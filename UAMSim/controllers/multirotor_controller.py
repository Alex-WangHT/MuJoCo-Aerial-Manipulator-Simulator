"""多旋翼控制器线程基类。

三线程模型（见 ``UAMSim.utils.frame_sync``）中的控制环一端：仿真线程每帧把
``SensorSnapshot`` 帧同步快照发布到 ``drone_channel`` 邮箱并阻塞等回执，
本线程 ``wait_snapshot`` 收到后调用用户重写的 ``controller()`` 计算推力，
写入旋翼执行器缓冲并 ``mark_done``，仿真线程随后在帧边界统一落盘 mjData。
本线程只读写通道对象，**绝不触碰 mjData**。

用法（用户唯一要重写的是 ``controller()``）::

    class HoverController(MultirotorController):
        def controller(self, feedback, target_z, hover):
            err = target_z - feedback.dronePosition[2]
            return np.full(self._multirotor.n_rotors, hover + 2.0 * err)

    sim.start(); sim.wait_ready()          # 编译完成、通道已暴露、物理仍停放
    ctrl = HoverController(sim.uam.multirotor, sim.drone_channel,
                           shutdown_event, target_z=1.0, hover=sim.uam.hover_thrust())
    ctrl.start()
    sim.start_physics()

``__init__`` 只接收绑定的 ``Multirotor`` 组件与 ``ControllerChannel``：
构造时自动注册控制输出（按 rotor1..N 为每个旋翼建 ``RotorActuator`` 缓冲并
``channel.attach``）与传感器反馈通路（订阅帧同步邮箱，快照逐帧传入
``controller()``）。``feedback`` 可用字段：``timestamp`` / ``timestep`` /
``dronePosition`` / ``droneVelocity`` / ``droneOrientation``（欧拉角）/
``droneAngularVelocity``——用什么做反馈由用户在 ``controller()`` 里自行决定。

``controller()`` 可定义自定义输入（目标参考、增益等）：构造函数里的多余关键字
参数原样存入 ``self.params``，每帧以 ``controller(feedback, **self.params)``
形式传入；不需要时签名即 ``controller(self, feedback)``。
"""

from __future__ import annotations

import threading

import numpy as np

from ..simulation.multirotor import Multirotor
from ..utils.actuators import RotorActuator
from ..utils.frame_sync import ControllerChannel
from ..utils.perception_bus import SensorSnapshot


class MultirotorController(threading.Thread):
    """多旋翼控制器线程基类（重写 :meth:`controller` 实现控制律）。

    参数:
        multirotor: 已绑定编译产物的 ``Multirotor`` 组件视图
            （``Environment.attach_robot()`` / ``sim.wait_ready()`` 之后）
        channel: 仿真线程暴露的 ``sim.drone_channel``
        shutdown_event: 外部共享退出事件（缺省内部自建；正常退出路径是仿真
            结束时通道 ``close()``，本事件只是额外保险）
        **controller_params: 自定义输入（参考、增益等），逐帧透传给
            ``controller(feedback, **params)``
    """

    def __init__(
        self,
        multirotor: Multirotor,
        channel: ControllerChannel,
        shutdown_event: threading.Event | None = None,
        **controller_params,
    ):
        super().__init__(daemon=True, name="MultirotorController")
        if not isinstance(multirotor, Multirotor):
            raise TypeError(
                f"MultirotorController: multirotor 应为 Multirotor 实例，"
                f"收到 {type(multirotor).__name__}"
            )
        if not isinstance(channel, ControllerChannel):
            raise TypeError(
                f"MultirotorController: channel 应为 ControllerChannel 实例，"
                f"收到 {type(channel).__name__}"
            )
        if multirotor.model is None or multirotor.data is None:
            raise RuntimeError(
                "MultirotorController: multirotor 尚未绑定编译产物，"
                "请在 Environment.attach_robot() / sim.wait_ready() 之后构造"
            )
        self._multirotor = multirotor
        self._channel = channel
        self._shutdown_event = shutdown_event if shutdown_event is not None else threading.Event()
        self.params = dict(controller_params)

        # ---- 自动注册控制输出：每个旋翼一个执行器缓冲，注册到通道 ----
        model, data = multirotor.model, multirotor.data
        self.rotors: list[RotorActuator] = [
            RotorActuator(
                data,
                multirotor.rotors[name],
                model.actuator_ctrlrange[multirotor.rotors[name]],
                initial=float(data.ctrl[multirotor.rotors[name]]),
                name=name,
            )
            for name in multirotor.rotor_names
        ]
        channel.attach(self.rotors)

        # ---- 可用传感器清单（MJCF 声明），反馈内容由用户在 controller() 里自取 ----
        self.sensor_names: list[str] = multirotor.sensor_names

    # ---------- 用户接口 ----------

    def controller(self, feedback: SensorSnapshot) -> np.ndarray:
        """用户重写的控制律：输入一帧反馈快照，返回 (n_rotors,) 推力 [N]。

        可定义自定义输入（参考、增益等），签名形如
        ``controller(self, feedback, target_pos, kp, ...)``——多余的关键字
        参数在构造时传入，逐帧透传。
        """
        raise NotImplementedError(
            "MultirotorController: 请在子类中重写 controller(feedback, ...)"
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
        n_rotors = self._multirotor.n_rotors
        last_frame = 0
        last_error = None
        while not self._shutdown_event.is_set():
            snapshot, frame = mailbox.wait_snapshot(last_frame)
            if snapshot is None:            # 通道关闭（仿真退出）
                break
            try:
                u = np.asarray(
                    self.controller(snapshot, **self.params), dtype=float
                ).reshape(-1)
                if u.size != n_rotors:
                    raise ValueError(
                        f"controller() 应返回 {n_rotors} 个推力值，收到 {u.size} 个"
                    )
                for rotor, value in zip(self.rotors, u):
                    rotor.set_thrust(float(value))
                last_error = None
            except Exception as exc:  # noqa: BLE001 - 用户控制律故障不应击落控制环
                message = f"{type(exc).__name__}: {exc}"
                if message != last_error:
                    print(f"[MultirotorController] controller() 异常（保持上一条指令）: {message}")
                    last_error = message
            mailbox.mark_done(frame)
            last_frame = frame
