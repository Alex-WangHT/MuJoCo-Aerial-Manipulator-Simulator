"""MuJoCo 空中机械臂仿真包 —— 除地面站外的全部仿真组件。

组件层（各自读取自身 MJCF，两阶段生命周期）:
- ``Multirotor``        多旋翼平台：读取 multirotor.xml，自省 actuator/sensor；
                        编译后 bind，读写接口 set_actuator / get_sensor
- ``Manipulator``       机械臂：读取 Manipulator.xml，同样模式

组合层:
- ``Environment``       MuJoCo 场景环境（地板/障碍物，唯一组合器：
                        attach_robot 接收组件实例，两级 attach 后合体编译）
- ``Robot``             已挂载机器人的视图句柄（attach_robot 返回值，
                        成组携带 multirotor/manipulator 组件视图）

线程/通信层（三线程模型）:
- ``MujocoSimulation``  仿真线程：输入 Environment + 机器人组件实例
                        （合并打包，仅对 Environment 编译），唯一访问 mjData；
                        编译后暴露 drone_channel / arm_channel 与 sim.uam 句柄
- ``MultirotorController``   多旋翼控制器线程基类；重写 control()
- ``ManipulatorController``  机械臂控制器线程基类；重写 compute_joint_targets()
- ``FrameSync``         FrameMailbox 帧同步邮箱 + ControllerChannel 通道
- ``Actuators``         RotorActuator / ServoActuator 执行器缓冲
                        （控制器写缓冲，仿真线程帧边界统一落盘 mjData）

数据载体:
- ``Messages``   SensorData / ManipulatorSensorData / ControlInput
- ``Telemetry``  TelemetryBuffer（线程安全遥测环形缓冲，GCS 读取端）
- ``TelemetryPublisher``  跨进程遥测通道（UDP+JSON 独立线程，非阻塞）

分层调用关系（Simulation 不接收 Controller 对象）::

    MujocoSimulation (threading.Thread)      仅编译 Environment，独占 mjData
      └── Environment              加载 models/environment.xml，统一编译
            └── attach_robot(Multirotor, Manipulator)  两级 attach，挂到 uam_spawn
                  ├── Multirotor   读 multirotor.xml（bind namespace="uam/"）
                  └── Manipulator  读 Manipulator.xml（bind namespace="uam/arm/")
                        （attach 返回值 Robot 句柄成组携带两组件视图）

    MultirotorController (thread) ──drone_channel──> 帧同步邮箱 + 旋翼执行器缓冲
    ManipulatorController (thread) ──arm_channel──> 帧同步邮箱 + 舵机执行器缓冲

地面站通过 TelemetryBuffer 与本包解耦。
"""

from __future__ import annotations

import math
import pathlib

import numpy as np

_MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"


def quaternionToEuler(q) -> np.ndarray:
    """w-x-y-z 四元数转 roll-pitch-yaw 欧拉角。"""
    w, x, y, z = np.asarray(q, dtype=float)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=float)


from .utils.Messages import ControlInput, ManipulatorSensorData, SensorData  # noqa: E402
from .utils.Telemetry import TelemetryBuffer  # noqa: E402
from .utils.TelemetryPublisher import TelemetryPublisher  # noqa: E402
from .utils.FrameSync import ControllerChannel, FrameMailbox  # noqa: E402
from .utils.Actuators import RotorActuator, ServoActuator  # noqa: E402
from .simulation.Multirotor import Multirotor  # noqa: E402
from .simulation.Manipulator import Manipulator  # noqa: E402
from .simulation.Robot import Robot  # noqa: E402
from .simulation.Environment import Environment  # noqa: E402
from .controller.MultirotorController import MultirotorController  # noqa: E402
from .controller.ManipulatorController import ManipulatorController  # noqa: E402
from .simulation.MujocoSimulation import MujocoSimulation  # noqa: E402

__all__ = [
    "Multirotor",
    "Manipulator",
    "Robot",
    "Environment",
    "MultirotorController",
    "ManipulatorController",
    "MujocoSimulation",
    "FrameMailbox",
    "ControllerChannel",
    "RotorActuator",
    "ServoActuator",
    "SensorData",
    "ManipulatorSensorData",
    "ControlInput",
    "TelemetryBuffer",
    "TelemetryPublisher",
    "quaternionToEuler",
]
