from __future__ import annotations

import threading

import numpy as np

from include.PIDController import PIDController


class DroneHeightController(threading.Thread):
    def __init__(
        self,
        shutdownEvent: threading.Event,
        heightControllerUpdateEvent: threading.Event,
        mass: float = 2.5,
        gravity: float = 9.81,
    ):
        super().__init__(daemon=True)
        self._shutdownEvent = shutdownEvent
        self._heightControllerUpdateEvent = heightControllerUpdateEvent
        self._lock = threading.Lock()
        self._actionLock = threading.Lock()
        self._heightData = {
            "timestamp": 0.0,
            "timestep": 0.001,
            "height": 0.0,
            "verticalVelocity": 0.0,
            "heightReference": 0.0,
        }
        self._heightPID = PIDController(kp=2.0, ki=0.0, kd=0.0, outputLimit=2.0)
        self._velocityPID = PIDController(
            kp=5.0,
            ki=1.0,
            kd=0.2,
            outputLimit=mass * gravity,
            integralLimit=3.0,
        )
        self._mass = mass
        self._gravity = gravity
        self._rotorForce = mass * gravity / 4.0
        self._reachedReference = False

    def run(self):
        print("[DroneHeightController] Initialized.")
        while not self._shutdownEvent.is_set():
            self.updateHeightController()
            with self._lock:
                data = dict(self._heightData)
            dt = max(float(data["timestep"]), 1e-6)
            height_error = data["heightReference"] - data["height"]
            velocity_reference = self._heightPID.compute(height_error, dt)
            velocity_error = velocity_reference - data["verticalVelocity"]
            collective = self._velocityPID.compute(velocity_error, dt)
            rotor_force = (self._mass * self._gravity + float(collective)) / 4.0
            with self._actionLock:
                self._rotorForce = max(0.0, rotor_force)
            self.checkReachedReferences(height_error, velocity_error)
        print("[DroneHeightController] Shutting down.")

    def stop(self) -> None:
        self._shutdownEvent.set()
        self._heightControllerUpdateEvent.set()

    def setHeightData(self, heightData: dict) -> None:
        with self._lock:
            self._heightData = heightData

    def updateHeightController(self) -> None:
        self._heightControllerUpdateEvent.wait(timeout=0.1)
        self._heightControllerUpdateEvent.clear()

    def checkReachedReferences(self, heightError, velocityError) -> None:
        self._reachedReference = abs(float(heightError)) < 0.03 and abs(float(velocityError)) < 0.05

    def getRotorForce(self) -> float:
        with self._actionLock:
            return float(self._rotorForce)

    def getReachedReference(self) -> bool:
        return self._reachedReference
