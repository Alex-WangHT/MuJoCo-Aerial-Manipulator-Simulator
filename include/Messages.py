from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Command:
    command: str
    height: float | None = None
    coordinates: tuple[float, float, float] | None = None
    attitude: tuple[float, float, float] | None = None
    attitudeOverride: bool | None = None
    q_combination_case: int = 0


@dataclass
class ControlMessage:
    referenceX: float | None = None
    referenceY: float | None = None
    referenceZ: float | None = None
    referenceRoll: float | None = None
    referencePitch: float | None = None
    referenceYaw: float | None = None
    attitudeOverride: bool = False
    yaw: float = 0.0
    grasp: bool = False
    qCombinationCase: int = 0


@dataclass
class SensorData:
    timestamp: float = 0.0
    timestep: float = 0.001
    dronePosition: np.ndarray = field(default_factory=lambda: np.zeros(3))
    droneVelocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    droneAcceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    droneOrientation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    droneAngularVelocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sensorValues: dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotSensorData:
    timestamp: float = 0.0
    timestep: float = 0.001
    manipulatorPosition: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dronePosition: np.ndarray = field(default_factory=lambda: np.zeros(3))
    jointsPosition: np.ndarray = field(default_factory=lambda: np.zeros(8))
    jointsVelocity: np.ndarray = field(default_factory=lambda: np.zeros(8))
    sensorValues: dict[str, Any] = field(default_factory=dict)
    qCombinationCase: int = 0


@dataclass
class ControlAction:
    droneThrusters: dict[str, float] = field(default_factory=dict)


@dataclass
class RobotControlAction:
    jointMotors: dict[str, float] = field(default_factory=dict)
