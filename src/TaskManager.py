from __future__ import annotations

import queue
import threading
import time

from include.Messages import Command, ControlMessage


class TaskManager(threading.Thread):
    def __init__(
        self,
        shutdownEvent: threading.Event,
        maneuverEvent: threading.Event,
        droneControlQueue: queue.Queue,
    ):
        super().__init__(daemon=True)
        self._shutdownEvent = shutdownEvent
        self._maneuverEvent = maneuverEvent
        self._droneControlQueue = droneControlQueue
        self._gcsMessageQueue: queue.Queue[Command] = queue.Queue()
        self._droneControlMessage = ControlMessage(grasp=False)

    def run(self):
        print("[TaskManager] Initialized.")
        while not self._shutdownEvent.is_set():
            try:
                message = self._gcsMessageQueue.get(timeout=0.05)
            except queue.Empty:
                continue
            self.executeCommand(message)
        print("[TaskManager] Shutting down.")

    def executeCommand(self, message: Command) -> None:
        print(f"[TaskManager] Executing: {message.command}")
        match message.command.lower():
            case "takeoff":
                self.takeOff(message.height if message.height is not None else 1.0)
            case "land":
                self.land()
            case "goto":
                if message.coordinates is None:
                    raise ValueError("goto command requires coordinates")
                self.goTo(message.coordinates)
            case "setpoint":
                self.setPoint(message.coordinates, message.attitude, message.attitudeOverride)
            case "grasp":
                self.grasp()
            case "release":
                self.release()
            case "sway":
                self._droneControlMessage.qCombinationCase = message.q_combination_case
                self.setDroneControlMessage()
            case _:
                print(f"[TaskManager] Unknown command: {message.command}")

    def takeOff(self, height: float) -> None:
        self._droneControlMessage.referenceZ = height
        self.setDroneControlMessage()
        print(f"[TaskManager] Taking off to {height:.2f} m.")

    def land(self) -> None:
        self._droneControlMessage.referenceZ = 0.0
        self.setDroneControlMessage()
        print("[TaskManager] Landing.")

    def goTo(self, coordinates: tuple[float, float, float]) -> None:
        self._droneControlMessage.referenceX = coordinates[0]
        self._droneControlMessage.referenceY = coordinates[1]
        self._droneControlMessage.referenceZ = coordinates[2]
        self.setDroneControlMessage()
        print(f"[TaskManager] Going to {coordinates}.")

    def setPoint(
        self,
        coordinates: tuple[float, float, float] | None,
        attitude: tuple[float, float, float] | None,
        attitudeOverride: bool | None,
    ) -> None:
        if coordinates is not None:
            self._droneControlMessage.referenceX = coordinates[0]
            self._droneControlMessage.referenceY = coordinates[1]
            self._droneControlMessage.referenceZ = coordinates[2]
        if attitude is not None:
            self._droneControlMessage.referenceRoll = attitude[0]
            self._droneControlMessage.referencePitch = attitude[1]
            self._droneControlMessage.referenceYaw = attitude[2]
            self._droneControlMessage.yaw = attitude[2]
            self._droneControlMessage.attitudeOverride = True
        if attitudeOverride is not None:
            self._droneControlMessage.attitudeOverride = attitudeOverride
        self.setDroneControlMessage()
        print(f"[TaskManager] Setpoint position={coordinates}, attitude={attitude}.")

    def grasp(self) -> None:
        self._droneControlMessage.grasp = True
        self.setDroneControlMessage()
        print("[TaskManager] Grasping object.")

    def release(self) -> None:
        self._droneControlMessage.grasp = False
        self.setDroneControlMessage()
        print("[TaskManager] Releasing object.")

    def setDroneControlMessage(self) -> None:
        self._maneuverEvent.set()
        self._droneControlQueue.put(self._droneControlMessage)

    def enqueue(self, command: Command) -> None:
        self._gcsMessageQueue.put(command)

    def stop(self) -> None:
        self._shutdownEvent.set()
        time.sleep(0.01)
