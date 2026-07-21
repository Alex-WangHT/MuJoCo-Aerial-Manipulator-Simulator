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


@dataclass
class ManipulatorSensorData:
    """机械臂传感快照（仿真线程每帧发布给机械臂控制器线程）。

    全部字段为当帧拷贝，控制器线程只读本快照、不触碰 mjData；
    ``eeJacobian`` 为末端位置雅可比在机械臂各关节 dof 上的切片
    (3, n_joints)，由仿真线程用 ``mj_jac`` 计算（雅可比依赖 mjData，
    不能在控制器线程内求取）。
    """

    timestamp: float = 0.0
    timestep: float = 0.001
    jointPositions: np.ndarray = field(default_factory=lambda: np.zeros(0))
    jointVelocities: np.ndarray = field(default_factory=lambda: np.zeros(0))
    eePosition: np.ndarray = field(default_factory=lambda: np.zeros(3))
    eeJacobian: np.ndarray = field(default_factory=lambda: np.zeros((3, 0)))
