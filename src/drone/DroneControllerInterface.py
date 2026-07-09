from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import threading

import numpy as np

from include.Messages import SensorData


@dataclass
class DroneFeedback:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    angularVelocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))


@dataclass
class DroneTargetPose:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))


@dataclass
class DroneControlInput:
    u: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))


class DroneControllerInterface(threading.Thread, ABC):
    def __init__(self, shutdownEvent: threading.Event):
        super().__init__(daemon=True)
        self._shutdownEvent = shutdownEvent
        self._feedbackLock = threading.Lock()
        self._targetLock = threading.Lock()
        self._controlLock = threading.Lock()
        self._updateEvent = threading.Event()

        self._feedback = DroneFeedback()
        self._targetPose = DroneTargetPose()
        self._controlInput = DroneControlInput()

    @abstractmethod
    def run(self) -> None:
        """Run the controller loop in its own thread."""

    def setFeedback(self, feedback: DroneFeedback) -> None:
        """Update the controller with the latest state feedback."""
        with self._feedbackLock:
            self._feedback = DroneFeedback(
                position=np.asarray(feedback.position, dtype=float).copy(),
                orientation=np.asarray(feedback.orientation, dtype=float).copy(),
                velocity=np.asarray(feedback.velocity, dtype=float).copy(),
                angularVelocity=np.asarray(feedback.angularVelocity, dtype=float).copy(),
            )

    def setTargetPose(self, targetPose: DroneTargetPose) -> None:
        """Update the desired position and attitude target."""
        with self._targetLock:
            self._targetPose = DroneTargetPose(
                position=np.asarray(targetPose.position, dtype=float).copy(),
                orientation=np.asarray(targetPose.orientation, dtype=float).copy(),
            )

    def setSensorData(self, sensorData: SensorData) -> None:
        """Compatibility helper for existing simulation code."""
        self.setFeedback(
            DroneFeedback(
                position=sensorData.dronePosition,
                orientation=sensorData.droneOrientation,
                velocity=sensorData.droneVelocity,
                angularVelocity=sensorData.droneAngularVelocity,
            )
        )

    def getFeedback(self) -> DroneFeedback:
        with self._feedbackLock:
            return DroneFeedback(
                position=self._feedback.position.copy(),
                orientation=self._feedback.orientation.copy(),
                velocity=self._feedback.velocity.copy(),
                angularVelocity=self._feedback.angularVelocity.copy(),
            )

    def getTargetPose(self) -> DroneTargetPose:
        with self._targetLock:
            return DroneTargetPose(
                position=self._targetPose.position.copy(),
                orientation=self._targetPose.orientation.copy(),
            )

    def setControlInput(self, controlInput: DroneControlInput) -> None:
        with self._controlLock:
            self._controlInput = DroneControlInput(u=np.asarray(controlInput.u, dtype=float).copy())

    def getControlInput(self) -> DroneControlInput:
        with self._controlLock:
            return DroneControlInput(u=self._controlInput.u.copy())

    def setUpdateEvent(self) -> None:
        self._updateEvent.set()

    def waitForUpdate(self, timeout: float = 0.05) -> bool:
        updated = self._updateEvent.wait(timeout=timeout)
        self._updateEvent.clear()
        return updated

    @abstractmethod
    def computeControl(self) -> DroneControlInput:
        """Compute the control input u from the stored feedback and target."""

    def reset(self) -> None:
        """Reset controller internal state such as integrators or filters."""
        with self._feedbackLock:
            self._feedback = DroneFeedback()
        with self._targetLock:
            self._targetPose = DroneTargetPose()
        with self._controlLock:
            self._controlInput = DroneControlInput()

    def stop(self) -> None:
        """Stop the controller thread and release any waiting loop."""
        self._shutdownEvent.set()
        self._updateEvent.set()
