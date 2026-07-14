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
