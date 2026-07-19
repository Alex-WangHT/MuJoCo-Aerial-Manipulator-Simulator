"""MuJoCo 空中机械臂仿真包 —— 除地面站外的全部仿真组件。

视图层:
- ``Multirotor``        多旋翼平台视图（旋翼/机体状态/IMU/挂载点/旋翼几何）
- ``Manipulator``       机械臂视图（关节/舵机/末端）

组合层:
- ``AerialManipulator`` 机器人组合器（平台 + 机械臂，mjSpec.attach）
- ``Environment``       MuJoCo 场景环境（地板/障碍物，最顶层组合器）

控制层（均为基类，用户继承后重写控制律方法即得自定义控制器）:
- ``MultirotorController``   默认串级 PID + 混控；重写 compute_control(sensor)->u
- ``ManipulatorController``  默认关节直通/末端 DLS；重写 compute_joint_targets()->q

仿真线程:
- ``MujocoSimulation``  主循环、viewer、遥测、控制器注入与绑定

数据载体:
- ``Messages``   SensorData / Command / ControlInput
- ``Telemetry``  TelemetryBuffer（线程安全遥测环形缓冲，GCS 读取端）

分层调用关系::

    MujocoSimulation (threading.Thread)
      ├── MultirotorController ──bind──> Multirotor 视图
      ├── ManipulatorController ─bind──> Manipulator 视图
      └── Environment              加载 models/environment.xml
            └── AerialManipulator  compileModel=False，挂到 uam_spawn
                  ├── Multirotor   视图（namespace="uam/"）
                  └── Manipulator  视图（namespace="uam/arm/"）

地面站通过 TelemetryBuffer 与本包解耦。
"""

from __future__ import annotations

import math
import pathlib

import numpy as np

_MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"


def quaternionToEuler(q) -> np.ndarray:
    """w-x-y-z 四元数转 roll-pitch-yaw 欧拉角。"""
    w, x, y, z = np.asarray(q, dtype=float)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=float)


from .Messages import Command, ControlInput, SensorData  # noqa: E402
from .Telemetry import TelemetryBuffer  # noqa: E402
from .Multirotor import Multirotor  # noqa: E402
from .Manipulator import Manipulator  # noqa: E402
from .AerialManipulator import AerialManipulator  # noqa: E402
from .Environment import Environment  # noqa: E402
from .MultirotorController import MultirotorController  # noqa: E402
from .ManipulatorController import ManipulatorController  # noqa: E402
from .MujocoSimulation import MujocoSimulation  # noqa: E402

__all__ = [
    "Multirotor",
    "Manipulator",
    "AerialManipulator",
    "Environment",
    "MultirotorController",
    "ManipulatorController",
    "MujocoSimulation",
    "SensorData",
    "Command",
    "ControlInput",
    "TelemetryBuffer",
    "quaternionToEuler",
]
