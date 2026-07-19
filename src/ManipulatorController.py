"""机械臂控制器基类（用户自定义控制器的父类）。

本类做两件事：

1. **协议与装配**：``bind(manipulator)`` 备好关节 dof 地址、关节范围与
   初始目标角；``update()`` 由仿真主循环每帧调用，取
   ``compute_joint_targets()`` 的结果写入位置舵机；
2. **默认控制律**：``compute_joint_targets()`` 支持两种模式——
   关节模式直接透传 ``set_target_joints`` 的目标角；
   末端模式以阻尼最小二乘（DLS）雅可比逆解朝 ``set_target_ee`` 的
   世界系目标点每帧迭代一步（步长限幅保证稳定）。

**用户自定义控制器**：继承本类，只重写
``compute_joint_targets() -> np.ndarray`` 一个方法即可——装配（bind）、
舵机写入与关节范围截断全部由基类完成。基类备好的
``_manipulator``（视图）、``_dof_adrs``、``_q_min/_q_max`` 可直接使用。

**稳定性保证（语法级）**：除 ``compute_joint_targets`` 外的全部基类方法
（``__init__`` / ``bind`` / ``update`` / 目标注入 / DLS 子步骤）已被冻结——
子类定义同名方法时类创建即抛 ``TypeError``，从语法上杜绝仿真协议被意外改写。
"""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None


class ManipulatorController:
    """机械臂控制器基类（默认：关节直通 / 末端 DLS 逆解）。"""

    # ---------- 子类化规则（语法级冻结） ----------
    # 用户自定义控制器只允许重写 _OVERRIDABLE_METHOD 一个方法；
    # 子类定义 _FROZEN_METHODS 中的同名方法时，类创建即抛 TypeError。
    _OVERRIDABLE_METHOD = "compute_joint_targets"
    _FROZEN_METHODS = frozenset({
        "__init__",
        "__init_subclass__",
        "_OVERRIDABLE_METHOD",
        "_FROZEN_METHODS",
        "bind",
        "set_target_joints",
        "set_target_ee",
        "mode",
        "update",
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
        jointTargets=None,
        eeDamping: float = 0.05,
        eeStepGain: float = 0.2,
        eeMaxStep: float = 0.02,
        eeTolerance: float = 1e-3,
    ):
        if mujoco is None:
            raise ImportError("ManipulatorController 需要 mujoco 包，请先 pip install mujoco")
        # 末端模式 DLS 参数（仅作用于默认实现）
        self._ee_damping = float(eeDamping)
        self._ee_step_gain = float(eeStepGain)   # 每帧梯度步增益（1 kHz 下已足够快）
        self._ee_max_step = float(eeMaxStep)     # 每帧关节目标最大变化 [rad]
        self._ee_tolerance = float(eeTolerance)

        # bind() 后填充（子类可直接使用的模型信息）
        self._bound = False
        self._manipulator = None                 # Manipulator 视图
        self._dof_adrs: list[int] = []           # 各关节 dof 地址（雅可比列索引）
        self._q_min: np.ndarray | None = None    # 关节下限 [rad]
        self._q_max: np.ndarray | None = None    # 关节上限 [rad]

        # 运行状态
        self._joint_targets = (
            None if jointTargets is None else np.asarray(jointTargets, dtype=float).reshape(-1)
        )
        self._ee_target: np.ndarray | None = None

    # ---------- 组装 ----------

    def bind(self, manipulator) -> None:
        """绑定机械臂视图，读取关节 dof 地址与关节范围。

        由 ``MujocoSimulation._build()`` 在模型编译完成后自动调用。
        未显式给过关节目标时，以保持当前关节角为初始目标。
        子类重写时通常应先调用 ``super().bind(manipulator)``。
        """
        self._manipulator = manipulator
        model = manipulator.model
        self._dof_adrs = [
            int(model.jnt_dofadr[manipulator.joints[name]])
            for name in manipulator.joint_names
        ]
        joint_ids = [manipulator.joints[name] for name in manipulator.joint_names]
        ranges = np.array([model.jnt_range[j] for j in joint_ids], dtype=float)
        self._q_min, self._q_max = ranges[:, 0], ranges[:, 1]

        if self._joint_targets is None:
            self._joint_targets = manipulator.get_joint_positions()
        if self._joint_targets.size != manipulator.n_joints:
            raise ValueError(
                f"ManipulatorController: 目标角数量 {self._joint_targets.size} "
                f"与关节数 {manipulator.n_joints} 不一致"
            )
        self._bound = True

    # ---------- 目标注入 ----------

    def set_target_joints(self, q) -> None:
        """关节模式：直接设置各关节目标角 [rad]。"""
        values = np.asarray(q, dtype=float).reshape(-1)
        if self._bound and values.size != self._manipulator.n_joints:
            raise ValueError(
                f"ManipulatorController: 期望 {self._manipulator.n_joints} 个目标角，"
                f"收到 {values.size} 个"
            )
        self._ee_target = None
        self._joint_targets = values

    def set_target_ee(self, position) -> None:
        """末端模式：设置末端执行器世界系目标位置 [x y z]。"""
        self._ee_target = np.asarray(position, dtype=float).reshape(3)

    @property
    def mode(self) -> str:
        return "ee" if self._ee_target is not None else "joints"

    # ---------- 主循环钩子（一般无需重写） ----------

    def update(self) -> None:
        """每帧调用：取控制律结果并写入舵机目标角。"""
        if not self._bound:
            raise RuntimeError(
                "ManipulatorController 尚未 bind(manipulator)，"
                "请将其传给 MujocoSimulation(manipulatorController=...) 由仿真自动绑定"
            )
        q = np.asarray(self.compute_joint_targets(), dtype=float).reshape(-1)
        if q.size != self._manipulator.n_joints:
            raise ValueError(
                f"compute_joint_targets 返回了 {q.size} 个目标角，"
                f"应为 {self._manipulator.n_joints} 个"
            )
        self._joint_targets = np.clip(q, self._q_min, self._q_max)
        self._manipulator.set_joint_targets(self._joint_targets)

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

        采用闭环逆运动学（CLIK）形式：在**实测关节角**上叠加修正量
        （而不是在上帧目标角上累加）。这样目标角最多领先物理状态一个
        步长限幅，舵机滞后时也不会形成目标-状态正反馈振荡——对含电枢
        惯量（armature）的高刚度舵机尤其关键。
        """
        arm = self._manipulator
        model, data = arm.model, arm.data

        ee_pos = data.site_xpos[arm.ee_site_id]
        err = self._ee_target - ee_pos
        q_meas = arm.get_joint_positions()
        if float(np.linalg.norm(err)) < self._ee_tolerance:
            return q_meas

        body_id = int(model.site_bodyid[arm.ee_site_id])
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, None, ee_pos.copy(), body_id)
        jacobian = jacp[:, self._dof_adrs]                     # (3, n_joints)

        # DLS: dq = Jᵀ (J Jᵀ + λ²I)⁻¹ err，增益缩放 + 每帧步长限幅
        dq = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + self._ee_damping ** 2 * np.eye(3), err
        )
        dq *= self._ee_step_gain
        dq = np.clip(dq, -self._ee_max_step, self._ee_max_step)
        return q_meas + dq
