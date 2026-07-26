"""uamsim 控制器子包。

提供多旋翼和机械臂的控制器线程基类：

- :class:`MultirotorController`：重写 ``controller()`` 实现旋翼控制律
- :class:`ManipulatorController`：重写 ``controller()`` 实现关节控制律
"""

from .multirotor_controller import MultirotorController
from .manipulator_controller import ManipulatorController

__all__ = ["MultirotorController", "ManipulatorController"]
