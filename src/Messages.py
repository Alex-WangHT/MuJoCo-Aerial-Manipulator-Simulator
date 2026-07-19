from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ControlInput:
    """控制器输出：旋翼推力向量 ``u`` [N]，维度 = 旋翼数。"""

    u: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class SensorData:
    timestamp: float = 0.0
    timestep: float = 0.001
    dronePosition: np.ndarray = field(default_factory=lambda: np.zeros(3))
    droneVelocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    droneOrientation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    droneAngularVelocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
