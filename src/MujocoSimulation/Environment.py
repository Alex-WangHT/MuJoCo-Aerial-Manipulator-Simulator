"""MuJoCo 场景环境。

加载环境 MJCF（地板 / 障碍物 / 出生点），把 AerialManipulator（未编译模式）
attach 进场景统一编译，是仿真模型的最顶层组合器；
提供障碍物查询与接触（碰撞）查询等交互接口。
"""

from __future__ import annotations

import pathlib

import numpy as np

try:
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None

from . import _MODELS_DIR
from .AerialManipulator import AerialManipulator


class Environment:
    """MuJoCo 场景环境：地板、灯光、障碍物与机器人的组合器。

    作为最顶层组合器加载 ``models/environment.xml``，机器人
    （``AerialManipulator``，需以 ``compileModel=False`` 构建）通过
    ``attach_robot()`` 挂到场景中的 ``uam_spawn`` site 上统一编译。
    编译后机器人的 ``Multirotor``/``Manipulator`` 视图自动绑定到场景
    共享模型（名称带机器人前缀，默认 ``uam/``）。

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

        self._attachments: list[tuple[AerialManipulator, str]] = []
        self.robots: list[AerialManipulator] = []

        self.model = None
        self.data = None
        self.obstacles: dict[str, int] = {}

    # ---------- 组合与编译 ----------

    def attach_robot(
        self,
        uam: AerialManipulator,
        site: str = SPAWN_SITE_NAME,
        prefix: str = "uam/",
    ) -> AerialManipulator:
        """将机器人挂到场景的 ``site`` 上并重新统一编译。

        ``uam`` 必须以 ``compileModel=False`` 构建；attach 后其全部名称
        获得 ``prefix`` 前缀（如 ``uam/drone``、``uam/arm/joint1``）。
        """
        if uam.model is not None:
            raise ValueError(
                "attach_robot 需要未编译的 AerialManipulator（compileModel=False）"
            )
        self._spec.attach(uam.spec, site=site, prefix=prefix)
        self._attachments.append((uam, prefix))
        self.compile()
        return uam

    def compile(self) -> None:
        """编译场景（含已挂载的全部机器人），并绑定各机器人视图。"""
        self.model = self._spec.compile()
        self.data = mujoco.MjData(self.model)

        self.robots = []
        for uam, prefix in self._attachments:
            uam.bind_views(self.model, self.data, namespace=prefix)
            self.robots.append(uam)

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

