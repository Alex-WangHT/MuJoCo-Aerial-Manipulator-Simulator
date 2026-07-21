"""机械臂控制器基类（独立控制线程，用户自定义控制器的父类）。

三线程模型下，控制器**不注入仿真**，而是在仿真编译完成后由外部
（如 main.py）构造并 ``start()``：

1. **装配**（构造时，物理尚未启动，读视图安全）：读取关节范围与初始
   目标角，按关节顺序实例化 :class:`~src.Actuators.ServoActuator` 并
   注册进 ``channel``；
2. **控制线程**：``run()`` 循环经帧同步邮箱接收仿真线程发布的
   :class:`~src.Messages.ManipulatorSensorData` 快照（含关节角/角速度/
   末端位置/末端雅可比切片，存入 ``self.sensor``），调用
   ``compute_joint_targets()``，把目标角写入舵机缓冲后回执——
   控制频率与物理帧严格同步（1 kHz）。

**用户自定义控制器**：继承本类，只重写
``compute_joint_targets() -> np.ndarray`` 一个方法即可——装配、
舵机写入与关节范围截断全部由基类完成。基类备好的 ``self.sensor``
（快照）、``_q_min/_q_max`` 可直接使用。

**稳定性保证（语法级）**：除 ``compute_joint_targets`` 外的全部基类
方法已被冻结——子类定义同名方法时类创建即抛 ``TypeError``。
"""

from __future__ import annotations

import threading
import traceback

import numpy as np

try:
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None

from .Actuators import ServoActuator
from .FrameSync import ControllerChannel
from .Messages import ManipulatorSensorData


class ManipulatorController(threading.Thread):
    """机械臂控制器线程基类（默认：关节直通 / 末端 DLS 逆解）。"""

    # ---------- 子类化规则（语法级冻结） ----------
    # 用户自定义控制器只允许重写 _OVERRIDABLE_METHOD 一个方法；
    # 子类定义 _FROZEN_METHODS 中的同名方法时，类创建即抛 TypeError。
    _OVERRIDABLE_METHOD = "compute_joint_targets"
    _FROZEN_METHODS = frozenset({
        "__init__",
        "__init_subclass__",
        "_OVERRIDABLE_METHOD",
        "_FROZEN_METHODS",
        "run",
        "set_target_joints",
        "set_target_ee",
        "mode",
        "_ee_dls_step",
    })

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        illegal = sorted(set(vars(cls)) & ManipulatorController._FROZEN_METHODS)
        if illegal:
            raise TypeError(
                f"{cls.__name__}: 为保证仿真链路稳定，控制器基类方法 {illegal} "
                "已冻结，不允许重写；用户自定义控制器只能重写 "
                f"'{ManipulatorController._OVERRIDABLE_METHOD}' 一个方法"
            )

    def __init__(
        self,
        channel: ControllerChannel,
        manipulator,
        jointTargets=None,
        eeDamping: float = 0.05,
        eeStepGain: float = 0.2,
        eeMaxStep: float = 0.02,
        eeTolerance: float = 1e-3,
    ):
        if mujoco is None:
            raise ImportError("ManipulatorController 需要 mujoco 包，请先 pip install mujoco")
        super().__init__(daemon=True, name="ManipulatorController")

        self._channel = channel

        # 末端模式 DLS 参数（仅作用于默认实现）
        self._ee_damping = float(eeDamping)
        self._ee_step_gain = float(eeStepGain)   # 每帧梯度步增益（1 kHz 下已足够快）
        self._ee_max_step = float(eeMaxStep)     # 每帧关节目标最大变化 [rad]
        self._ee_tolerance = float(eeTolerance)

        # ---- 装配（构造时物理尚未启动，读 mjData 安全）----
        self._n_joints = manipulator.n_joints
        joint_ids = [manipulator.joints[name] for name in manipulator.joint_names]
        ranges = np.array([manipulator.model.jnt_range[j] for j in joint_ids], dtype=float)
        self._q_min, self._q_max = ranges[:, 0], ranges[:, 1]

        if jointTargets is None:
            jointTargets = manipulator.get_joint_positions()
        self._joint_targets = np.asarray(jointTargets, dtype=float).reshape(-1)
        if self._joint_targets.size != self._n_joints:
            raise ValueError(
                f"ManipulatorController: 目标角数量 {self._joint_targets.size} "
                f"与关节数 {self._n_joints} 不一致"
            )
        self._joint_targets = np.clip(self._joint_targets, self._q_min, self._q_max)
        self._ee_target: np.ndarray | None = None

        # ---- 自动实例化舵机执行器并注册进通道 ----
        self.actuators: list[ServoActuator] = [
            ServoActuator(
                manipulator.data,
                manipulator.servos[name],
                ranges[i],
                self._joint_targets[i],
                name=name,
            )
            for i, name in enumerate(manipulator.joint_names)
        ]
        channel.attach(self.actuators)

        self.sensor = ManipulatorSensorData()  # 最新快照（首帧发布后填充）

    # ---------- 目标注入 ----------

    def set_target_joints(self, q) -> None:
        """关节模式：直接设置各关节目标角 [rad]。"""
        values = np.asarray(q, dtype=float).reshape(-1)
        if values.size != self._n_joints:
            raise ValueError(
                f"ManipulatorController: 期望 {self._n_joints} 个目标角，收到 {values.size} 个"
            )
        self._ee_target = None
        self._joint_targets = np.clip(values, self._q_min, self._q_max)

    def set_target_ee(self, position) -> None:
        """末端模式：设置末端执行器世界系目标位置 [x y z]。"""
        self._ee_target = np.asarray(position, dtype=float).reshape(3)

    @property
    def mode(self) -> str:
        return "ee" if self._ee_target is not None else "joints"

    # ---------- 线程入口（已冻结） ----------

    def run(self) -> None:
        """控制循环：等快照 -> compute_joint_targets() -> 写舵机缓冲 -> 回执；
        邮箱关闭即退出。控制律抛异常不杀线程（打印堆栈前 3 次），
        沿用执行器缓冲中的上一帧指令。
        """
        mailbox = self._channel.mailbox
        last_frame = -1
        failures = 0
        while True:
            snapshot, frame = mailbox.wait_snapshot(last_frame)
            if snapshot is None:
                break
            last_frame = frame
            self.sensor = snapshot
            try:
                q = np.asarray(self.compute_joint_targets(), dtype=float).reshape(-1)
                if q.size != self._n_joints:
                    raise ValueError(
                        f"compute_joint_targets 返回了 {q.size} 个目标角，应为 {self._n_joints} 个"
                    )
                self._joint_targets = np.clip(q, self._q_min, self._q_max)
                for actuator, value in zip(self.actuators, self._joint_targets):
                    actuator.set_target(float(value))
            except Exception:
                failures += 1
                if failures <= 3:
                    print(f"[ManipulatorController] compute_joint_targets() 第 {failures} 次异常：")
                    traceback.print_exc()
            mailbox.mark_done(frame)

    # ---------- 控制律（用户重写点） ----------

    def compute_joint_targets(self) -> np.ndarray:
        """计算本帧关节目标角 [rad]。

        **用户自定义控制器只需重写本方法。** 默认实现：关节模式透传目标角；
        末端模式朝 EE 目标做一次 DLS 雅可比迭代（步长限幅）。
        """
        if self._ee_target is not None:
            return self._ee_dls_step()
        return self._joint_targets

    # ---------- 内部（默认实现的子步骤，子类可复用） ----------

    def _ee_dls_step(self) -> np.ndarray:
        """朝末端目标做一次阻尼最小二乘雅可比迭代（每帧单步保证稳定）。

        采用闭环逆运动学（CLIK）形式：在**实测关节角**（快照）上叠加修正量
        （而不是在上帧目标角上累加）。这样目标角最多领先物理状态一个
        步长限幅，舵机滞后时也不会形成目标-状态正反馈振荡——对含电枢
        惯量（armature）的高刚度舵机尤其关键。

        雅可比取自快照 ``self.sensor.eeJacobian``（仿真线程用 mj_jac 算好
        随帧发布），控制器线程不触碰 mjData。
        """
        sensor = self.sensor
        err = self._ee_target - sensor.eePosition
        q_meas = np.asarray(sensor.jointPositions, dtype=float)
        if float(np.linalg.norm(err)) < self._ee_tolerance:
            return q_meas

        jacobian = np.asarray(sensor.eeJacobian, dtype=float)      # (3, n_joints)
        # DLS: dq = Jᵀ (J Jᵀ + λ²I)⁻¹ err，增益缩放 + 每帧步长限幅
        dq = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + self._ee_damping ** 2 * np.eye(3), err
        )
        dq *= self._ee_step_gain
        dq = np.clip(dq, -self._ee_max_step, self._ee_max_step)
        return q_meas + dq
