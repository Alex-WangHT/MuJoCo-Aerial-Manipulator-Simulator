"""``src`` 命名空间 —— 兼容转发层。

全部仿真组件位于 :mod:`src.MujocoSimulation` 包；本模块只做再导出，
保证 ``from src import MujocoSimulation`` 等旧写法继续可用。
新代码请直接从包根导入::

    from src.MujocoSimulation import MujocoSimulation, TelemetryBuffer
"""

from __future__ import annotations

from .MujocoSimulation import (
    AerialManipulator,
    Command,
    ControlInput,
    Environment,
    Manipulator,
    ManipulatorController,
    MujocoSimulation,
    Multirotor,
    MultirotorController,
    SensorData,
    TelemetryBuffer,
    quaternionToEuler,
)

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
