"""空中机械臂组合器。

把 Multirotor 与 Manipulator 两个组件（各自已读取自身 MJCF）打包成
整机机器人：机械臂 spec 经 ``mjSpec.attach`` 固连到多旋翼 spec 的挂载
site 上，编译后回调两个组件的 ``bind()`` 生成视图；也支持
``compileModel=False`` 的未编译模式，供 Environment 二次组合统一编译。
"""

from __future__ import annotations

try:
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None

from .Manipulator import Manipulator
from .Multirotor import Multirotor


class AerialManipulator:
    """空中机械臂（UAM）机器人组合器。

    职责：
    - 接收已读取 MJCF 的 ``Multirotor`` / ``Manipulator`` 组件实例，
      以平台 MJCF 中声明的挂载点 site（默认 ``manipulator_mount``，位于
      机体系）作为固连位置，通过 ``mjSpec.attach`` 把机械臂 spec 挂进
      平台 spec，组合出整机机器人 spec；
    - ``manipulator=None`` 时为纯多旋翼构型：不 attach，
      ``has_manipulator`` 为 ``False``；
    - 两种使用方式：

      1. 独立运行（默认 ``compileModel=True``）：立即编译出共享
         ``MjModel``/``MjData``，机械臂与平台动力学天然耦合；
      2. 放入场景（``compileModel=False``）：只构建 ``self.spec``，由
         ``Environment.attach_robot()`` 挂到场景统一编译，之后通过
         ``bind_views()`` 绑定到场景模型（名称自动带场景前缀）。

    构造示例::

        uam = AerialManipulator(Multirotor(), Manipulator())                 # 独立编译
        uam = AerialManipulator(Multirotor(), Manipulator(), compileModel=False)  # 交给场景
        uam = AerialManipulator(Multirotor())                                # 纯多旋翼
    """

    def __init__(
        self,
        multirotor: Multirotor,
        manipulator: Manipulator | None = None,
        mountSite: str = Multirotor.MOUNT_SITE_NAME,
        armPrefix: str = "arm/",
        compileModel: bool = True,
    ):
        if mujoco is None:
            raise ImportError("AerialManipulator 需要 mujoco 包，请先 pip install mujoco")
        if not isinstance(multirotor, Multirotor):
            raise TypeError(
                f"AerialManipulator: multirotor 应为 Multirotor 实例，收到 {type(multirotor).__name__}"
            )
        if manipulator is not None and not isinstance(manipulator, Manipulator):
            raise TypeError(
                f"AerialManipulator: manipulator 应为 Manipulator 实例或 None，"
                f"收到 {type(manipulator).__name__}"
            )

        # ---- 组件（已各自读取 MJCF） ----
        self.multirotor = multirotor
        self.manipulator = manipulator

        # ---- 打包：机械臂 spec 固连到平台挂载点，子模型名称自动加前缀 ----
        self.spec = self.multirotor.spec
        self._arm_prefix = armPrefix if manipulator is not None else None
        self._attach_frame = (
            self.spec.attach(manipulator.spec, site=mountSite, prefix=armPrefix)
            if manipulator is not None
            else None
        )

        self.model = None
        self.data = None

        if compileModel:
            self.compile()

    # ---------- 编译与视图绑定 ----------

    def compile(self) -> None:
        """独立模式：编译自身 spec 并绑定无前缀视图。"""
        model = self.spec.compile()
        data = mujoco.MjData(model)
        self.bind_views(model, data, namespace="")
        # 前向一次运动学/传感器，使初始位姿与 sensordata 立即可读
        mujoco.mj_forward(self.model, self.data)

    def bind_views(self, model, data, namespace: str = "") -> None:
        """将 Multirotor / Manipulator 视图绑定到（可能是场景共享的）模型。

        由独立模式的 ``compile()`` 或 ``Environment.attach_robot()`` 调用；
        ``namespace`` 为场景 attach 时使用的前缀（如 ``uam/``）。
        纯多旋翼构型（未传 ``manipulator``）下 ``manipulator`` 为 ``None``。
        """
        self.model = model
        self.data = data
        self.multirotor.bind(model, data, namespace)
        if self.manipulator is not None:
            self.manipulator.bind(model, data, namespace + self._arm_prefix)

    @property
    def has_manipulator(self) -> bool:
        """当前构型是否包含机械臂。"""
        return self.manipulator is not None

    def _require_compiled(self) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError(
                "AerialManipulator 尚未编译：请使用 compileModel=True，"
                "或通过 Environment.attach_robot() 挂到场景统一编译"
            )

    # ---------- 仿真推进 ----------

    def step(self, n_steps: int = 1) -> None:
        self._require_compiled()
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def reset(self) -> None:
        """重置动力学状态。注意：若模型由 Environment 共享，会重置整个场景。"""
        self._require_compiled()
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    # ---------- 便捷量 ----------

    def hover_thrust(self) -> float:
        """整机悬停所需的单旋翼推力 [N]（总质量 × g / 旋翼数）。"""
        self._require_compiled()
        total_mass = float(self.model.body_mass.sum())
        return total_mass * 9.81 / self.multirotor.n_rotors
