from __future__ import annotations

import threading
import time

import numpy as np

from include.Messages import RobotControlAction, RobotSensorData
from include.PIDController import PIDController


JOINT_NAMES = [
    "left_shoulder_roll",
    "left_shoulder_pitch",
    "left_elbow_roll",
    "left_elbow_pitch",
    "right_shoulder_roll",
    "right_shoulder_pitch",
    "right_elbow_roll",
    "right_elbow_pitch",
]


class SwayControl(threading.Thread):
    def __init__(self, shutdownEvent: threading.Event):
        super().__init__(daemon=True)
        self._shutdownEvent = shutdownEvent
        self._sensorLock = threading.Lock()
        self._controlLock = threading.Lock()
        self._updateEvent = threading.Event()
        self._sensorData = RobotSensorData()
        self._robotControlAction = RobotControlAction(jointMotors={})
        self._positionPID = PIDController(
            kp=np.array([1, 1, 12, 12, 1, 1, 12, 12], dtype=float),
            ki=np.array([0, 0, 5, 5, 0, 0, 5, 5], dtype=float),
            kd=np.zeros(8),
            outputLimit=1.0,
            integralLimit=np.array([0, 0, 0.3, 0.3, 0, 0, 0.3, 0.3], dtype=float),
        )
        self._velocityPID = PIDController(
            kp=np.array([1, 1, 10, 10, 1, 1, 10, 10], dtype=float),
            ki=np.array([1, 1, 5, 5, 1, 1, 5, 5], dtype=float),
            kd=np.array([0, 0, 0, 0.01, 0, 0, 0, 0.01], dtype=float),
            outputLimit=1.0,
            integralLimit=np.array([0.5, 0.5, 0.3, 0.3, 0.5, 0.5, 0.3, 0.3], dtype=float),
        )
        self._refFilter = FirstOrderRef(alpha=0.1)
        self._velLPF = LowPass(alpha=0.15)
        self._targetQPosition = np.zeros(8)

    def run(self):
        print("[SwayControl] Initialized.")
        previous_pos_timestamp = 0.0
        previous_vel_timestamp = 0.0
        while not self._shutdownEvent.is_set():
            self.updateCurrentState()
            with self._sensorLock:
                sensor = self._sensorData

            self._targetQPosition = self.targetForCase(sensor.qCombinationCase)
            target_q = self._refFilter.step(self._targetQPosition)
            pos_dt = max(sensor.timestamp - previous_pos_timestamp, 0.005)
            qd_reference = self._positionPID.compute(target_q - sensor.jointsPosition, pos_dt)
            previous_pos_timestamp = sensor.timestamp

            vel_dt = max(sensor.timestamp - previous_vel_timestamp, 0.001)
            q_velocity_filtered = self._velLPF.filt(sensor.jointsVelocity)
            torque = self._velocityPID.compute(qd_reference - q_velocity_filtered, vel_dt)
            previous_vel_timestamp = sensor.timestamp

            with self._controlLock:
                self._robotControlAction = RobotControlAction(
                    jointMotors={name: float(torque[i]) for i, name in enumerate(JOINT_NAMES)}
                )
        print("[SwayControl] Shutting down.")

    def stop(self) -> None:
        self._shutdownEvent.set()
        self._updateEvent.set()

    def updateCurrentState(self) -> None:
        self._updateEvent.wait(timeout=0.1)
        self._updateEvent.clear()

    def setCorrespondingTargetQPosition(self) -> None:
        with self._sensorLock:
            self._targetQPosition = self.targetForCase(self._sensorData.qCombinationCase)

    @staticmethod
    def targetForCase(qCombinationCase: int) -> np.ndarray:
        match qCombinationCase:
            case 1:
                return np.array([0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0, 1.57])
            case 2:
                return np.array([1.57, 0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0])
            case 3:
                return np.array([0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0, 0.0])
            case 4:
                return np.array([1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            case 5:
                return np.array([0.0, 1.57, 0.0, 0.0, 0.0, -1.57, 0.0, 0.0])
            case 6:
                return np.array([0.0, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            case 7:
                return np.array([1.57, 0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0])
            case _:
                return np.zeros(8)

    def setSensorData(self, sensorData: RobotSensorData) -> None:
        with self._sensorLock:
            self._sensorData = sensorData

    def getControlInputs(self) -> RobotControlAction:
        with self._controlLock:
            return self._robotControlAction

    def setUpdateEvent(self) -> None:
        self._updateEvent.set()


class LowPass:
    def __init__(self, alpha=0.15):
        self.y = None
        self.alpha = alpha

    def filt(self, x):
        x = np.asarray(x, dtype=float)
        if self.y is None:
            self.y = x.copy()
        self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y


class FirstOrderRef:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.y = None

    def step(self, r):
        r = np.asarray(r, dtype=float)
        if self.y is None:
            self.y = r.copy()
        self.y = self.y + self.alpha * (r - self.y)
        return self.y
