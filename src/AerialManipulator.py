"""空中机械臂组合器。

用 mjSpec.attach 把机械臂 MJCF 固连到多旋翼 MJCF 的挂载 site 上，
编译后生成 Multirotor / Manipulator 两个带名称前缀的视图；
也支持 compileModel=False 的未编译模式，供 Environment 二次组合。
"""

from __future__ import annotations

import pathlib

try:
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None

from . import _MODELS_DIR
from .Manipulator import Manipulator
from .Multirotor import Multirotor


class AerialManipulator:
    """空中机械臂（UAM）机器人组合器。

    职责：
    - 加载多旋翼平台 MJCF（``multirotor.xml``）；``withManipulator=True``（默认）
      时再加载机械臂 MJCF（``Manipulator.xml``），以平台 MJCF 中声明的挂载点
      site（默认 ``manipulator_mount``，位于机体系）作为固连位置，通过
      ``mjSpec.attach`` 组合机器人 spec；
    - ``withManipulator=False`` 时为纯多旋翼构型：不加载机械臂，
      编译后 ``manipulator`` 视图为 ``None``；
    - 两种使用方式：

      1. 独立运行（默认 ``compileModel=True``）：立即编译出共享
         ``MjModel``/``MjData``，机械臂与平台动力学天然耦合；
      2. 放入场景（``compileModel=False``）：只构建 ``self.spec``，由
         ``Environment.attach_robot()`` 挂到场景统一编译，之后通过
         ``bind_views()`` 绑定到场景模型（名称自动带场景前缀）。
    """

    def __init__(
        self,
        multirotorPath: str | None = None,
        manipulatorPath: str | None = None,
        mountSite: str = Multirotor.MOUNT_SITE_NAME,
        armPrefix: str = "arm/",
        compileModel: bool = True,
        withManipulator: bool = True,
    ):
        if mujoco is None:
            raise ImportError("AerialManipulator 需要 mujoco 包，请先 pip install mujoco")

        platform_path = pathlib.Path(multirotorPath) if multirotorPath else _MODELS_DIR / "multirotor.xml"
        platform_spec = mujoco.MjSpec.from_file(str(platform_path))

        # 挂载点位姿声明在 multirotor.xml 的 drone 机体系内，attach 后机械臂
        # worldbody 转换为固连在该 site 上的 frame，子模型名称自动加前缀
        self._arm_prefix = armPrefix if withManipulator else None
        if withManipulator:
            arm_path = pathlib.Path(manipulatorPath) if manipulatorPath else _MODELS_DIR / "Manipulator.xml"
            arm_spec = mujoco.MjSpec.from_file(str(arm_path))
            self._attach_frame = platform_spec.attach(arm_spec, site=mountSite, prefix=armPrefix)
        else:
            self._attach_frame = None

        self.spec = platform_spec
        self.model = None
        self.data = None
        self.multirotor: Multirotor | None = None
        self.manipulator: Manipulator | None = None

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
        纯多旋翼构型（``withManipulator=False``）下 ``manipulator`` 为 ``None``。
        """
        self.model = model
        self.data = data
        self.multirotor = Multirotor(model, data, namespace)
        self.manipulator = (
            Manipulator(model, data, namespace + self._arm_prefix)
            if self._arm_prefix is not None
            else None
        )

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

