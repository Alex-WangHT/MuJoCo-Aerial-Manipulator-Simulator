"""已挂载机器人的视图句柄。

``Robot`` 由 ``Environment.attach_robot()`` 返回（也存于 ``Environment.robots``），
把绑定到场景共享模型上的 ``Multirotor`` / ``Manipulator`` 组件视图成组携带，
并提供整机便捷量（如悬停推力）。spec 组合与编译全部由 ``Environment`` 完成，
本类不含任何组合/编译逻辑。
"""

from __future__ import annotations

from .Manipulator import Manipulator
from .Multirotor import Multirotor


class Robot:
    """已挂载机器人的视图句柄（``Environment.attach_robot()`` 的返回类型）。

    属性：
    - ``multirotor`` / ``manipulator``：绑定到场景共享模型的组件视图
      （纯多旋翼构型下 ``manipulator`` 为 ``None``）
    - ``prefix``：本机器人在场景中的名称前缀（如 ``uam/``）
    """

    def __init__(
        self,
        multirotor: Multirotor,
        manipulator: Manipulator | None,
        prefix: str,
    ):
        self.multirotor = multirotor
        self.manipulator = manipulator
        self.prefix = prefix

    @property
    def has_manipulator(self) -> bool:
        """当前构型是否包含机械臂。"""
        return self.manipulator is not None

    @property
    def model(self):
        """组件绑定的场景共享 MjModel（未绑定时为 None）。"""
        return self.multirotor.model

    @property
    def data(self):
        """组件绑定的场景共享 MjData（未绑定时为 None）。"""
        return self.multirotor.data

    def hover_thrust(self) -> float:
        """整机悬停所需的单旋翼推力 [N]（总质量 × g / 旋翼数）。"""
        if self.model is None or self.data is None:
            raise RuntimeError(
                "Robot 尚未绑定编译产物：请经 Environment.attach_robot() 挂载编译后再调用"
            )
        total_mass = float(self.model.body_mass.sum())
        return total_mass * 9.81 / self.multirotor.n_rotors
