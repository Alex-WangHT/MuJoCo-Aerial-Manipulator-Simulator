"""uamsim 工具子包。

提供帧同步、执行器缓冲、遥测和感知等辅助组件：

- :class:`FrameMailbox` / :class:`ControllerChannel`：仿真线程与控制器线程的帧同步通道
- :class:`RotorActuator` / :class:`ServoActuator`：执行器缓冲
- :class:`TelemetryBuffer` / :class:`TelemetryPublisher`：遥测数据缓存与发布
- :class:`SensorSnapshot` / :class:`PerceptionSource`：传感快照与感知源
"""

from .actuators import RotorActuator, ServoActuator
from .frame_sync import ControllerChannel, FrameMailbox
from .perception_bus import (
    PerceptionPublisher,
    PerceptionReceiver,
    PerceptionSource,
    SensorSnapshot,
)
from .telemetry import TelemetryBuffer
from .telemetry_publisher import TelemetryPublisher

__all__ = [
    "ControllerChannel",
    "FrameMailbox",
    "RotorActuator",
    "ServoActuator",
    "SensorSnapshot",
    "PerceptionSource",
    "PerceptionPublisher",
    "PerceptionReceiver",
    "TelemetryBuffer",
    "TelemetryPublisher",
]
