from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from include.PIDController import PIDController
from src.drone.DroneControllerInterface import (
    DroneControlInput,
    DroneControllerInterface,
    DroneFeedback,
    DroneTargetPose,
)


@dataclass
class CascadedPIDConfig:
    mass: float = 1.74
    gravity: float = 9.81
    rotorCount: int = 6
    minRotorThrust: float = 0.0
    maxRotorThrust: float = 8.0
    maxTiltRad: float = 0.35
    positionKp: np.ndarray = field(default_factory=lambda: np.array([1.2, 1.2, 2.0], dtype=float))
    positionKi: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    positionKd: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    positionLimit: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.5], dtype=float))
    velocityKp: np.ndarray = field(default_factory=lambda: np.array([1.8, 1.8, 4.0], dtype=float))
    velocityKi: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.4], dtype=float))
    velocityKd: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, 0.4], dtype=float))
    velocityLimit: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 5.0], dtype=float))
    velocityIntegralLimit: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 1.0], dtype=float))
    attitudeKp: np.ndarray = field(default_factory=lambda: np.array([6.0, 6.0, 2.0], dtype=float))
    attitudeKi: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    attitudeKd: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    attitudeLimit: np.ndarray = field(default_factory=lambda: np.array([3.0, 3.0, 1.0], dtype=float))
    rateKp: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.25, 0.08], dtype=float))
    rateKi: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.02, 0.01], dtype=float))
    rateKd: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.01, 0.002], dtype=float))
    rateLimit: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 1.0], dtype=float))
    rateIntegralLimit: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.25], dtype=float))


class CascadedPIDDroneController(DroneControllerInterface):
    def __init__(
        self,
        shutdownEvent: threading.Event,
        config: CascadedPIDConfig | None = None,
    ):
        super().__init__(shutdownEvent)
        self._config = config or CascadedPIDConfig()
        self._positionPID = PIDController(
            kp=self._config.positionKp,
            ki=self._config.positionKi,
            kd=self._config.positionKd,
            outputLimit=self._config.positionLimit,
        )
        self._velocityPID = PIDController(
            kp=self._config.velocityKp,
            ki=self._config.velocityKi,
            kd=self._config.velocityKd,
            outputLimit=self._config.velocityLimit,
            integralLimit=self._config.velocityIntegralLimit,
        )
        self._attitudePID = PIDController(
            kp=self._config.attitudeKp,
            ki=self._config.attitudeKi,
            kd=self._config.attitudeKd,
            outputLimit=self._config.attitudeLimit,
        )
        self._ratePID = PIDController(
            kp=self._config.rateKp,
            ki=self._config.rateKi,
            kd=self._config.rateKd,
            outputLimit=self._config.rateLimit,
            integralLimit=self._config.rateIntegralLimit,
        )
        self._lastUpdateTime: float | None = None

    def run(self) -> None:
        print("[CascadedPIDDroneController] Initialized.")
        while not self._shutdownEvent.is_set():
            updated = self.waitForUpdate(timeout=0.05)
            if not updated and self._lastUpdateTime is None:
                continue
            self.setControlInput(self.computeControl())
        print("[CascadedPIDDroneController] Shutting down.")

    def computeControl(self) -> DroneControlInput:
        feedback = self.getFeedback()
        target = self.getTargetPose()
        dt = self._computeDt()

        desired_velocity = self._positionPID.compute(target.position - feedback.position, dt)
        desired_acceleration = self._velocityPID.compute(desired_velocity - feedback.velocity, dt)

        desired_attitude = self._computeDesiredAttitude(desired_acceleration, target.orientation)
        attitude_error = np.array(
            [
                wrap_angle(desired_attitude[0] - feedback.orientation[0]),
                wrap_angle(desired_attitude[1] - feedback.orientation[1]),
                wrap_angle(desired_attitude[2] - feedback.orientation[2]),
            ],
            dtype=float,
        )
        desired_rate = self._attitudePID.compute(attitude_error, dt)
        rate_error = desired_rate - feedback.angularVelocity
        roll_u, pitch_u, yaw_u = self._ratePID.compute(rate_error, dt)

        collective_per_rotor = self._computeCollectivePerRotor(desired_acceleration[2], feedback, desired_attitude)
        rotor_adjustments = self._mixHexX(roll_u, pitch_u, yaw_u)
        rotor_thrust = np.clip(
            collective_per_rotor + rotor_adjustments,
            self._config.minRotorThrust,
            self._config.maxRotorThrust,
        )
        return DroneControlInput(u=rotor_thrust)

    def reset(self) -> None:
        super().reset()
        self._positionPID.reset()
        self._velocityPID.reset()
        self._attitudePID.reset()
        self._ratePID.reset()
        self._lastUpdateTime = None

    def _computeDt(self) -> float:
        now = time.perf_counter()
        if self._lastUpdateTime is None:
            self._lastUpdateTime = now
            return 0.01
        dt = max(now - self._lastUpdateTime, 1e-3)
        self._lastUpdateTime = now
        return dt

    def _computeDesiredAttitude(
        self,
        desired_acceleration: np.ndarray,
        target_orientation: np.ndarray,
    ) -> np.ndarray:
        yaw_reference = float(target_orientation[2])
        ax, ay, _ = np.asarray(desired_acceleration, dtype=float)
        gravity = self._config.gravity

        roll_from_position = (ax * math.sin(yaw_reference) - ay * math.cos(yaw_reference)) / gravity
        pitch_from_position = -(ax * math.cos(yaw_reference) + ay * math.sin(yaw_reference)) / gravity

        roll_target = np.clip(
            roll_from_position + float(target_orientation[0]),
            -self._config.maxTiltRad,
            self._config.maxTiltRad,
        )
        pitch_target = np.clip(
            pitch_from_position + float(target_orientation[1]),
            -self._config.maxTiltRad,
            self._config.maxTiltRad,
        )
        return np.array([roll_target, pitch_target, yaw_reference], dtype=float)

    def _computeCollectivePerRotor(
        self,
        desired_vertical_acceleration: float,
        feedback: DroneFeedback,
        desired_attitude: np.ndarray,
    ) -> float:
        # Compensate part of the tilt loss so altitude hold does not collapse while rolling/pitching.
        roll = float(feedback.orientation[0])
        pitch = float(feedback.orientation[1])
        desired_roll = float(desired_attitude[0])
        desired_pitch = float(desired_attitude[1])
        effective_roll = 0.5 * (roll + desired_roll)
        effective_pitch = 0.5 * (pitch + desired_pitch)
        tilt_compensation = max(math.cos(effective_roll) * math.cos(effective_pitch), 0.4)

        total_thrust = self._config.mass * (self._config.gravity + float(desired_vertical_acceleration))
        total_thrust /= tilt_compensation
        return float(total_thrust / self._config.rotorCount)

    @staticmethod
    def _mixHexX(roll_u: float, pitch_u: float, yaw_u: float) -> np.ndarray:
        return np.array(
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


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
