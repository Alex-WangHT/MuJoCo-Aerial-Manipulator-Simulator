from __future__ import annotations

import math
import threading

import numpy as np

from include.PIDController import PIDController


class DroneStabilityController(threading.Thread):
    def __init__(
        self,
        shutdownEvent: threading.Event,
        stabilityOrientationUpdateEvent: threading.Event,
    ):
        super().__init__(daemon=True)
        self._shutdownEvent = shutdownEvent
        self._stabilityOrientationUpdateEvent = stabilityOrientationUpdateEvent
        self._lock = threading.Lock()
        self._actionLock = threading.Lock()
        self._stabilityData = {
            "timestamp": 0.0,
            "timestep": 0.001,
            "currentOrientation": np.zeros(3),
            "orientationReference": np.zeros(3),
            "currentAngularVelocity": np.zeros(3),
        }
        self._anglePID = PIDController(kp=np.array([6.0, 6.0, 2.0]), outputLimit=np.array([3.0, 3.0, 1.0]))
        self._ratePID = PIDController(
            kp=np.array([0.25, 0.25, 0.08]),
            ki=np.array([0.0, 0.0, 0.0]),
            kd=np.array([0.01, 0.01, 0.002]),
            outputLimit=np.array([2.0, 2.0, 1.0]),
            integralLimit=np.array([0.5, 0.5, 0.25]),
        )
        self._rotorAdjustments = np.zeros(6)

    def run(self):
        print("[DroneStabilityController] Initialized.")
        while not self._shutdownEvent.is_set():
            self.updateOrientation()
            with self._lock:
                data = dict(self._stabilityData)
            dt = max(float(data["timestep"]), 1e-6)
            angle_error = wrap_angles(data["orientationReference"] - data["currentOrientation"])
            target_rate = self._anglePID.compute(angle_error, dt)
            rate_error = target_rate - data["currentAngularVelocity"]
            roll_u, pitch_u, yaw_u = self._ratePID.compute(rate_error, dt)
            rotor_adjustments = np.array(
                [
                    -pitch_u + yaw_u,
                    +0.866 * roll_u - 0.5 * pitch_u - yaw_u,
                    +0.866 * roll_u + 0.5 * pitch_u + yaw_u,
                    +pitch_u - yaw_u,
                    -0.866 * roll_u + 0.5 * pitch_u + yaw_u,
                    -0.866 * roll_u - 0.5 * pitch_u - yaw_u,
                ],
                dtype=float,
            )
            with self._actionLock:
                self._rotorAdjustments = rotor_adjustments
        print("[DroneStabilityController] Shutting down.")

    def updateOrientation(self) -> None:
        self._stabilityOrientationUpdateEvent.wait(timeout=0.1)
        self._stabilityOrientationUpdateEvent.clear()

    def setStabilityData(self, stabilityData: dict) -> None:
        with self._lock:
            self._stabilityData = stabilityData

    def getRotorAdjustments(self) -> np.ndarray:
        with self._actionLock:
            return self._rotorAdjustments.copy()

    def stop(self) -> None:
        self._shutdownEvent.set()
        self._stabilityOrientationUpdateEvent.set()


def wrap_angles(angles):
    return (np.asarray(angles, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi


def quaternionToEuler(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])
