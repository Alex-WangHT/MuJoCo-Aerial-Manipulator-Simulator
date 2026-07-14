from __future__ import annotations

import math

import numpy as np

from include.Messages import SensorData


def quaternionToEuler(q):
    w, x, y, z = np.asarray(q, dtype=float)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=float)


class Robot:
    """无人机状态提取层。

    负责从 MuJoCo ``model`` / ``data`` 中提取无人机位姿、速度、姿态等
    传感器数据，并将控制器输出的执行器指令写入 ``data.ctrl``。
    """

    def __init__(self, model, data):
        self._model = model
        self._data = data

    def getSensorData(self) -> SensorData:
        """从 MuJoCo ``qpos`` / ``qvel`` 提取无人机状态。"""
        timestamp = float(self._data.time)
        qpos = self._data.qpos
        qvel = self._data.qvel

        drone_position = np.array(qpos[:3]) if qpos.size >= 3 else np.zeros(3)
        drone_velocity = np.array(qvel[:3]) if qvel.size >= 3 else np.zeros(3)
        drone_orientation = quaternionToEuler(qpos[3:7]) if qpos.size >= 7 else np.zeros(3)
        drone_angular_velocity = np.array(qvel[3:6]) if qvel.size >= 6 else np.zeros(3)

        return SensorData(
            timestamp=timestamp,
            timestep=self._model.opt.timestep,
            dronePosition=drone_position,
            droneVelocity=drone_velocity,
            droneOrientation=drone_orientation,
            droneAngularVelocity=drone_angular_velocity,
        )

    def applyControl(self, controlDict: dict[str, float]) -> None:
        """将控制器输出的执行器指令写入 MuJoCo ``data.ctrl``。

        如果某个 actuator 名称在模型中不存在，则静默跳过。
        """
        for actuator_name, value in controlDict.items():
            try:
                actuator_id = self._model.actuator(actuator_name).id
                self._data.ctrl[actuator_id] = value
            except Exception:
                continue
