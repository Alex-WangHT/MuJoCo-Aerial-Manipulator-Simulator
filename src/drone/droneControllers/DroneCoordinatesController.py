from __future__ import annotations

import threading

import numpy as np

from include.PIDController import PIDController


class CoordinatesController(threading.Thread):
    def __init__(
        self,
        shutdownEvent: threading.Event,
        coordinatesUpdateEvent: threading.Event,
        gravity: float = 9.81,
        mass: float = 2.5,
        rotorCount: int = 6,
    ):
        super().__init__(daemon=True)
        self._shutdownEvent = shutdownEvent
        self._coordinatesUpdateEvent = coordinatesUpdateEvent
        self._gravity = gravity
        self._mass = mass
        self._rotorCount = rotorCount
        self._lock = threading.Lock()
        self._actionLock = threading.Lock()
        self._coordinatesData = {
            "timestamp": 0.0,
            "timestep": 0.001,
            "currentPosition": np.zeros(3),
            "currentVelocity": np.zeros(3),
            "positionReference": np.zeros(3),
            "yawReference": 0.0,
        }
        self._positionPID = PIDController(
            kp=np.array([1.2, 1.2, 2.0]),
            ki=np.zeros(3),
            kd=np.zeros(3),
            outputLimit=np.array([1.0, 1.0, 1.5]),
        )
        self._velocityPID = PIDController(
            kp=np.array([1.8, 1.8, 4.0]),
            ki=np.array([0.0, 0.0, 0.0]),
            kd=np.array([0.1, 0.1, 0.4]),
            outputLimit=np.array([2.0, 2.0, 5.0]),
            integralLimit=np.array([0.5, 0.5, 0.5]),
        )
        self._attitudeReference = np.zeros(3)
        self._rotorForce = mass * gravity / rotorCount
        self._reachedReference = False

    def run(self):
        print("[DroneCoordinatesController] Initialized.")
        while not self._shutdownEvent.is_set():
            self.updateState()
            with self._lock:
                data = dict(self._coordinatesData)
            dt = max(float(data["timestep"]), 1e-6)
            position_error = data["positionReference"] - data["currentPosition"]
            velocity_reference = self._positionPID.compute(position_error, dt)
            velocity_error = velocity_reference - data["currentVelocity"]
            desired_acceleration = self._velocityPID.compute(velocity_error, dt)
            reference_euler = self.computeAttitude(desired_acceleration)
            reference_euler[2] = data["yawReference"]
            rotor_force = self.computeRotorForce(desired_acceleration[2])
            with self._actionLock:
                self._attitudeReference = reference_euler
                self._rotorForce = rotor_force
            self.checkReachedReferences(position_error, velocity_error)
        print("[DroneCoordinatesController] Shutting down.")

    def stop(self) -> None:
        self._shutdownEvent.set()
        self._coordinatesUpdateEvent.set()

    def computeAttitude(self, desiredAcceleration):
        ax, ay, _ = np.asarray(desiredAcceleration, dtype=float)
        roll = np.clip(ay / self._gravity, -0.35, 0.35)
        pitch = np.clip(-ax / self._gravity, -0.35, 0.35)
        return np.array([roll, pitch, 0.0], dtype=float)

    def computeRotorForce(self, desiredVerticalAcceleration):
        total_thrust = self._mass * (self._gravity + float(desiredVerticalAcceleration))
        rotor_force = total_thrust / self._rotorCount
        return float(np.clip(rotor_force, 0.0, self._mass * self._gravity))

    def updateState(self) -> None:
        self._coordinatesUpdateEvent.wait(timeout=0.1)
        self._coordinatesUpdateEvent.clear()

    def checkReachedReferences(self, positionError, velocityError) -> None:
        position_norm = np.linalg.norm(np.asarray(positionError, dtype=float))
        velocity_norm = np.linalg.norm(np.asarray(velocityError, dtype=float))
        self._reachedReference = position_norm < 0.05 and velocity_norm < 0.05

    def setCoordinatesData(self, coordinatesData: dict) -> None:
        with self._lock:
            self._coordinatesData = coordinatesData

    def getAttitudeReference(self) -> np.ndarray:
        with self._actionLock:
            return self._attitudeReference.copy()

    def getRotorForce(self) -> float:
        with self._actionLock:
            return float(self._rotorForce)

    def getReachedReference(self) -> bool:
        return self._reachedReference

    @staticmethod
    def normalizeVector(vector):
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if norm < 1e-9:
            return vector
        return vector / norm
