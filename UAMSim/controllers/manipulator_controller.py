"""机械臂控制器线程基类。

三线程模型（见 ``UAMSim.utils.frame_sync``）中的控制环一端：仿真线程每帧把
``SensorSnapshot`` 帧同步快照发布到 ``arm_channel`` 邮箱并阻塞等回执，
本线程 ``wait_snapshot`` 收到后调用用户重写的 ``controller()`` 计算关节
目标角，写入舵机执行器缓冲并 ``mark_done``，仿真线程随后在帧边界统一落盘
mjData。本线程只读写通道对象，**绝不触碰 mjData**。

用法（用户唯一要重写的是 ``controller()``）::

    class HoldController(ManipulatorController):
        def controller(self, feedback, q_target):
            return q_target

    sim.start(); sim.wait_ready()          # 编译完成、通道已暴露、物理仍停放
    ctrl = HoldController(sim.uam.manipulator, sim.arm_channel,
                          shutdown_event,
                          q_target=sim.uam.manipulator.get_joint_positions())
    ctrl.start()
    sim.start_physics()

``__init__`` 只接收绑定的 ``Manipulator`` 组件与 ``ControllerChannel``：
构造时自动注册控制输出（按运动树序为每个关节建 ``ServoActuator`` 缓冲并
``channel.attach``）与传感器反馈通路（订阅帧同步邮箱，快照逐帧传入
``controller()``）。``feedback`` 可用字段：``timestamp`` / ``timestep`` /
``jointPositions`` / ``jointVelocities`` / ``eePosition`` / ``eeJacobian``
——用什么做反馈由用户在 ``controller()`` 里自行决定。

``controller()`` 可定义自定义输入（目标参考、增益等）：构造函数里的多余关键字
参数原样存入 ``self.params``，每帧以 ``controller(feedback, **self.params)``
形式传入；不需要时签名即 ``controller(self, feedback)``。
"""

from __future__ import annotations

import threading

import numpy as np

from ..simulation.manipulator import Manipulator
from ..utils.actuators import ServoActuator
from ..utils.frame_sync import ControllerChannel
from ..utils.perception_bus import SensorSnapshot


class ManipulatorController(threading.Thread):
    """机械臂控制器线程基类（重写 :meth:`controller` 实现控制律）。

    参数:
        manipulator: 已绑定编译产物的 ``Manipulator`` 组件视图
            （``Environment.attach_robot()`` / ``sim.wait_ready()`` 之后）
        channel: 仿真线程暴露的 ``sim.arm_channel``
        shutdown_event: 外部共享退出事件（缺省内部自建；正常退出路径是仿真
            结束时通道 ``close()``，本事件只是额外保险）
        **controller_params: 自定义输入（参考、增益等），逐帧透传给
            ``controller(feedback, **params)``
    """

    def __init__(
        self,
        manipulator: Manipulator,
        channel: ControllerChannel,
        shutdown_event: threading.Event | None = None,
        **controller_params,
    ):
        super().__init__(daemon=True, name="ManipulatorController")
        if not isinstance(manipulator, Manipulator):
            raise TypeError(
                f"ManipulatorController: manipulator 应为 Manipulator 实例，"
                f"收到 {type(manipulator).__name__}"
            )
        if not isinstance(channel, ControllerChannel):
            raise TypeError(
                f"ManipulatorController: channel 应为 ControllerChannel 实例，"
                f"收到 {type(channel).__name__}"
            )
        if manipulator.model is None or manipulator.data is None:
            raise RuntimeError(
                "ManipulatorController: manipulator 尚未绑定编译产物，"
                "请在 Environment.attach_robot() / sim.wait_ready() 之后构造"
            )
        self._manipulator = manipulator
        self._channel = channel
        self._shutdown_event = shutdown_event if shutdown_event is not None else threading.Event()
        self.params = dict(controller_params)

        # ---- 自动注册控制输出：每个关节一个舵机缓冲，注册到通道 ----
        model, data = manipulator.model, manipulator.data
        self.servos: list[ServoActuator] = [
            ServoActuator(
                data,
                manipulator.servos[name],
                model.actuator_ctrlrange[manipulator.servos[name]],
                initial=float(data.ctrl[manipulator.servos[name]]),
                name=name,
            )
            for name in manipulator.joint_names
        ]
        channel.attach(self.servos)

        # ---- 可用传感器清单（MJCF 声明），反馈内容由用户在 controller() 里自取 ----
        self.sensor_names: list[str] = manipulator.sensor_names

    # ---------- 用户接口 ----------

    def controller(self, feedback: SensorSnapshot) -> np.ndarray:
        """用户重写的控制律：输入一帧反馈快照，返回 (n_joints,) 关节目标角 [rad]。

        可定义自定义输入（参考、增益等），签名形如
        ``controller(self, feedback, q_target, kp, ...)``——多余的关键字
        参数在构造时传入，逐帧透传。
        """
        raise NotImplementedError(
            "ManipulatorController: 请在子类中重写 controller(feedback, ...)"
        )

    # ---------- 继承约束 ----------

    def __init_subclass__(cls, **kwargs) -> None:
        """语法层面封死 ``__init__`` / ``run``：子类只允许重写 ``controller()``。

        自定义状态（参考、增益、滤波器等）一律经构造 kwargs 进入 ``self.params``，
        由 ``controller()`` 的自定义参数接收，不需要也不允许重写构造函数。
        """
        super().__init_subclass__(**kwargs)
        for sealed in ("__init__", "run"):
            if sealed in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__}: 不允许重写 {sealed}()，"
                    "用户接口只有 controller()；自定义参数请用构造 kwargs"
                )

    # ---------- 线程入口 ----------

    def run(self) -> None:
        """帧同步控制环：等快照 -> 计算 -> 写缓冲 -> 回执。

        ``controller()`` 抛异常时保持上一条缓冲指令、照常回执（避免仿真线程
        死等在 ``wait_done``），错误内容变化时打印一次。
        """
        mailbox = self._channel.mailbox
        n_joints = self._manipulator.n_joints
        last_frame = 0
        last_error = None
        while not self._shutdown_event.is_set():
            snapshot, frame = mailbox.wait_snapshot(last_frame)
            if snapshot is None:            # 通道关闭（仿真退出）
                break
            try:
                q = np.asarray(
                    self.controller(snapshot, **self.params), dtype=float
                ).reshape(-1)
                if q.size != n_joints:
                    raise ValueError(
                        f"controller() 应返回 {n_joints} 个目标角，收到 {q.size} 个"
                    )
                for servo, value in zip(self.servos, q):
                    servo.set_target(float(value))
                last_error = None
            except Exception as exc:  # noqa: BLE001 - 用户控制律故障不应击落控制环
                message = f"{type(exc).__name__}: {exc}"
                if message != last_error:
                    print(f"[ManipulatorController] controller() 异常（保持上一条指令）: {message}")
                    last_error = message
            mailbox.mark_done(frame)
            last_frame = frame
