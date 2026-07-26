"""MuJoCo 空中机械臂仿真包。

通过 pip install -e . 安装后，即可::

    import UAMSim as ua

    # 从外部 XML 创建环境、多旋翼与机械臂
    env = ua.Environment("path/to/environment.xml")
    uam = env.attach_robot(
        ua.Multirotor("path/to/multirotor.xml"),
        ua.Manipulator("path/to/manipulator.xml"),
    )

    # 创建仿真线程
    sim = ua.MujocoSimulation(shutdown_event, env,
                               ua.Multirotor("path/to/multirotor.xml"),
                               ua.Manipulator("path/to/manipulator.xml"))
    sim.start()
    sim.wait_ready()

    # 自定义控制器（只需重写 controller 方法）
    class MyDroneCtrl(ua.MultirotorController):
        def controller(self, feedback, kp):
            err = self.params["target"] - feedback.dronePosition
            return self.uam.hover_thrust() + kp * err[2]

    ctrl = MyDroneCtrl(sim.uam.multirotor, sim.drone_channel,
                        target=[0,0,1.5], kp=2.0)
    ctrl.start()
    sim.start_physics()
"""

from __future__ import annotations

import math

import numpy as np


def quaternion_to_euler(q) -> np.ndarray:
    """w-x-y-z 四元数转 roll-pitch-yaw 欧拉角 [rad]。"""
    w, x, y, z = np.asarray(q, dtype=float)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=float)


# 保留旧名兼容（未来版本可能弃用）
quaternionToEuler = quaternion_to_euler


# ---------------------------------------------------------------------------
# 公共 API 导出
# ---------------------------------------------------------------------------

# simulation
from .simulation.environment import Environment  # noqa: E402
from .simulation.multirotor import Multirotor  # noqa: E402
from .simulation.manipulator import Manipulator  # noqa: E402
from .simulation.robot import Robot  # noqa: E402
from .simulation.mujoco_simulation import MujocoSimulation  # noqa: E402

# controllers
from .controllers.multirotor_controller import MultirotorController  # noqa: E402
from .controllers.manipulator_controller import ManipulatorController  # noqa: E402

# utils
from .utils.frame_sync import ControllerChannel, FrameMailbox  # noqa: E402
from .utils.actuators import RotorActuator, ServoActuator  # noqa: E402
from .utils.perception_bus import (  # noqa: E402
    PerceptionPublisher,
    PerceptionReceiver,
    PerceptionSource,
    SensorSnapshot,
)
from .utils.telemetry import TelemetryBuffer  # noqa: E402
from .utils.telemetry_publisher import TelemetryPublisher  # noqa: E402


__version__ = "0.1.0"
__all__ = [
    # simulation
    "Environment",
    "Multirotor",
    "Manipulator",
    "Robot",
    "MujocoSimulation",
    # controllers
    "MultirotorController",
    "ManipulatorController",
    # utils / sync
    "ControllerChannel",
    "FrameMailbox",
    "RotorActuator",
    "ServoActuator",
    "SensorSnapshot",
    # utils / telemetry & perception
    "TelemetryBuffer",
    "TelemetryPublisher",
    "PerceptionSource",
    "PerceptionPublisher",
    "PerceptionReceiver",
    # helpers
    "quaternion_to_euler",
    "quaternionToEuler",
]
