"""MuJoCo 场景环境。

加载环境 MJCF（地板 / 障碍物 / 出生点），把机器人组件（Multirotor + 可选
Manipulator）经两级 mjSpec attach（机械臂→挂载点、整机→出生点）组合进场景
统一编译，是仿真模型的唯一组合器；
提供障碍物查询与接触（碰撞）查询等交互接口。
"""

from __future__ import annotations

import pathlib

import numpy as np

try:
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None

from .. import _MODELS_DIR
from .Manipulator import Manipulator
from .Multirotor import Multirotor
from .Robot import Robot


class Environment:
    """MuJoCo 场景环境：地板、灯光、障碍物与机器人的组合器。

    作为唯一组合器加载 ``models/environment.xml``，机器人组件
    （``Multirotor`` + 可选 ``Manipulator``，均已各自读取 MJCF、尚未绑定）
    通过 ``attach_robot()`` 完成组合：机械臂 spec 固连到多旋翼的
    ``manipulator_mount`` site（名称加 ``armPrefix``），整机再挂到场景的
    ``uam_spawn`` site 上（名称加 ``prefix``，默认 ``uam/``），随后统一编译。
    编译后组件视图自动绑定到场景共享模型，返回 ``Robot`` 句柄。

    交互查询：
    - ``obstacles`` / ``get_obstacle_positions()``：障碍物名称与位置
    - ``get_contacts()``：当前接触对（机器人 vs 场景/障碍物的物理交互）
    """

    SPAWN_SITE_NAME = "uam_spawn"
    OBSTACLE_PREFIX = "obstacle_"

    def __init__(self, environmentPath: str | None = None):
        if mujoco is None:
            raise ImportError("Environment 需要 mujoco 包，请先 pip install mujoco")

        env_path = pathlib.Path(environmentPath) if environmentPath else _MODELS_DIR / "environment.xml"
        self._spec = mujoco.MjSpec.from_file(str(env_path))

        self._attachments: list[tuple[Multirotor, Manipulator | None, str, str]] = []
        self.robots: list[Robot] = []

        self.model = None
        self.data = None
        self.obstacles: dict[str, int] = {}

    # ---------- 组合与编译 ----------

    def attach_robot(
        self,
        multirotor: Multirotor,
        manipulator: Manipulator | None = None,
        site: str = SPAWN_SITE_NAME,
        prefix: str = "uam/",
        mountSite: str = Multirotor.MOUNT_SITE_NAME,
        armPrefix: str = "arm/",
    ) -> Robot:
        """组合机器人组件并挂到场景的 ``site`` 上，重新统一编译。

        两级 attach：有机械臂时先把 ``manipulator`` 的 spec 固连到多旋翼的
        ``mountSite``（子模型名称加 ``armPrefix``），再把整机 spec 挂到场景
        ``site``（全部名称加 ``prefix``，如 ``uam/drone``、``uam/arm/joint1``）。

        组件须尚未绑定（一经 attach 编译即绑定到场景，不可重复挂载）。
        返回绑定完成的 ``Robot`` 句柄（同入 ``self.robots``）。
        """
        if not isinstance(multirotor, Multirotor):
            raise TypeError(
                f"attach_robot: multirotor 应为 Multirotor 实例，收到 {type(multirotor).__name__}"
            )
        if manipulator is not None and not isinstance(manipulator, Manipulator):
            raise TypeError(
                f"attach_robot: manipulator 应为 Manipulator 实例或 None，"
                f"收到 {type(manipulator).__name__}"
            )
        if multirotor.model is not None or (manipulator is not None and manipulator.model is not None):
            raise ValueError(
                "attach_robot 需要未绑定的组件：组件一经 attach 编译即绑定到场景，"
                "请重新构造 Multirotor / Manipulator 实例"
            )

        # ---- 第一级：机械臂 spec 固连到平台挂载点，子模型名称加 arm 前缀 ----
        if manipulator is not None:
            multirotor.spec.attach(manipulator.spec, site=mountSite, prefix=armPrefix)

        # ---- 第二级：整机 spec 挂到场景出生点，全部名称加机器人前缀 ----
        self._spec.attach(multirotor.spec, site=site, prefix=prefix)
        self._attachments.append((multirotor, manipulator, prefix, armPrefix))
        self.compile()
        return self.robots[-1]

    def compile(self) -> None:
        """编译场景（含已挂载的全部机器人），并绑定各机器人组件视图。"""
        self.model = self._spec.compile()
        self.data = mujoco.MjData(self.model)

        self.robots = []
        for multirotor, manipulator, prefix, armPrefix in self._attachments:
            multirotor.bind(self.model, self.data, prefix)
            if manipulator is not None:
                manipulator.bind(self.model, self.data, prefix + armPrefix)
            self.robots.append(Robot(multirotor, manipulator, prefix))

        self._refresh_obstacles()
        # 前向一次运动学，使位姿/接触/传感器立即可读
        mujoco.mj_forward(self.model, self.data)

    def _refresh_obstacles(self) -> None:
        self.obstacles = {}
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith(self.OBSTACLE_PREFIX):
                self.obstacles[name] = geom_id

    def _require_compiled(self) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError("Environment 尚未编译：请先 attach_robot() 或调用 compile()")

    # ---------- 仿真推进 ----------

    def step(self, n_steps: int = 1) -> None:
        self._require_compiled()
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def reset(self) -> None:
        self._require_compiled()
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    # ---------- 场景交互查询 ----------

    def get_obstacle_positions(self) -> dict[str, np.ndarray]:
        """各障碍物的世界坐标。"""
        self._require_compiled()
        return {
            name: self.data.geom_xpos[geom_id].copy()
            for name, geom_id in self.obstacles.items()
        }

    def get_contacts(self) -> list[tuple[str, str, np.ndarray]]:
        """当前全部接触对：(geom1 名称, geom2 名称, 接触点世界坐标)。"""
        self._require_compiled()
        contacts = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or f"geom{contact.geom1}"
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or f"geom{contact.geom2}"
            contacts.append((name1, name2, np.array(contact.pos, dtype=float)))
        return contacts

    def get_robot_contacts(self, prefix: str = "uam/") -> list[tuple[str, str, np.ndarray]]:
        """只保留涉及指定机器人（按名称前缀）的接触对。"""
        return [
            (name1, name2, pos)
            for name1, name2, pos in self.get_contacts()
            if name1.startswith(prefix) or name2.startswith(prefix)
        ]

