from __future__ import annotations

import argparse
import threading
import tkinter as tk
from collections.abc import Callable

import numpy as np

from include.Telemetry import TelemetryBuffer
from src.GroundControlStation import GCS
from src.MujocoSimulation import MujocoSimulation
from src.drone import DroneControlInput, DroneControllerInterface


FIXED_THRUST_PER_ROTOR = 3


class OpenLoopFixedThrustController(DroneControllerInterface):
    def __init__(
        self,
        shutdownEvent: threading.Event,
        rotorCount: int = 6,
        fixedThrustPerRotor: float = FIXED_THRUST_PER_ROTOR,
    ):
        super().__init__(shutdownEvent)
        self._rotorCount = rotorCount
        self._fixedThrustPerRotor = fixedThrustPerRotor

    def run(self) -> None:
        print("[OpenLoopFixedThrustController] Initialized.")
        while not self._shutdownEvent.is_set():
            self.waitForUpdate(timeout=0.05)
            self.setControlInput(self.computeControl())
        print("[OpenLoopFixedThrustController] Shutting down.")

    def computeControl(self) -> DroneControlInput:
        u = np.full(self._rotorCount, max(0.0, self._fixedThrustPerRotor), dtype=float)
        return DroneControlInput(u=u)


def buildDefaultDroneController(shutdownEvent: threading.Event) -> DroneControllerInterface:
    return OpenLoopFixedThrustController(shutdownEvent)


def main(
    droneControllerFactory: Callable[[threading.Event], DroneControllerInterface] | None = None,
) -> None:
    parser = argparse.ArgumentParser(description="DroneControllerInterface MuJoCo test.")
    parser.add_argument("--model", default="models/common_uam.xml")
    parser.add_argument("--nogui", action="store_true", help="Run without GCS telemetry window.")
    args = parser.parse_args()

    shutdownEvent = threading.Event()
    telemetryBuffer = TelemetryBuffer()
    controllerFactory = droneControllerFactory or buildDefaultDroneController
    droneController = controllerFactory(shutdownEvent)
    mujocoSim = MujocoSimulation(
        shutdownEvent,
        droneController=droneController,
        modelPath=args.model,
        telemetryBuffer=telemetryBuffer,
    )

    print(
        f"[Main] Using drone controller: {type(droneController).__name__} "
        f"with fixed thrust {FIXED_THRUST_PER_ROTOR:.2f} N per rotor."
    )
    droneController.start()
    mujocoSim.start()

    try:
        if args.nogui:
            while mujocoSim.is_alive() and not shutdownEvent.is_set():
                mujocoSim.join(timeout=0.25)
        else:
            root = tk.Tk()
            GCS(root, taskManager=None, shutdownEvent=shutdownEvent, telemetryBuffer=telemetryBuffer)
            root.mainloop()
    except KeyboardInterrupt:
        shutdownEvent.set()
    finally:
        mujocoSim.stop()
        if mujocoSim.is_alive():
            mujocoSim.join(timeout=1.0)
        droneController.stop()
        if droneController.is_alive():
            droneController.join(timeout=1.0)


if __name__ == "__main__":
    main()
