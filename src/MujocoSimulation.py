from __future__ import annotations

import pathlib
import math
import threading
import time

import numpy as np

from cython_modules.fast_mujoco_utils import extract_data
from include.Messages import RobotSensorData, SensorData
from include.Telemetry import TelemetryBuffer
from src.drone.DroneControllerInterface import DroneControllerInterface
from src.manipulator.SwayControl import JOINT_NAMES, SwayControl

try:
    import mujoco
    import mujoco.viewer
except Exception:  # pragma: no cover - allows framework use before installing mujoco.
    mujoco = None


class MujocoSimulation(threading.Thread):
    def __init__(
        self,
        shutdownEvent: threading.Event,
        droneController: DroneControllerInterface | None = None,
        swayController: SwayControl | None = None,
        swayControllerEnabled: bool = False,
        modelPath: str | None = None,
        timestep: float = 0.001,
        telemetryBuffer: TelemetryBuffer | None = None,
        fixedRotorThrust: float | None = None,
    ):
        super().__init__(daemon=True)
        self._shutdownEvent = shutdownEvent
        self._droneController = droneController
        self._swayController = swayController
        self._swayControllerEnabled = swayControllerEnabled
        self._timestep = timestep
        self._modelPath = pathlib.Path(modelPath or "models/UAM_1cable.xml")
        self._telemetryBuffer = telemetryBuffer
        self._fixedRotorThrust = fixedRotorThrust
        self._model = None
        self._data = None

    def run(self):
        print("[MujocoSimulation] Initialized.")
        if mujoco is None:
            print("[MujocoSimulation] mujoco package not installed; running clock-only sensor loop.")
            self._runWithoutMujoco()
            return

        self._model = mujoco.MjModel.from_xml_path(str(self._modelPath))
        self._data = mujoco.MjData(self._model)
        self.printBalanceDiagnostics()
        self.launchViewer()
        print("[MujocoSimulation] Ending simulation.")

    def _runWithoutMujoco(self) -> None:
        timestamp = 0.0
        position = np.zeros(3)
        velocity = np.zeros(3)
        while not self._shutdownEvent.is_set():
            sensor = SensorData(timestamp=timestamp, timestep=self._timestep, dronePosition=position, droneVelocity=velocity)
            if self._telemetryBuffer is not None:
                self._telemetryBuffer.append(timestamp, position, np.zeros(3))
            robot = RobotSensorData(timestamp=timestamp, timestep=self._timestep, dronePosition=position)
            if self._droneController is not None:
                self._droneController.setSensorData(sensor)
                self._droneController.setUpdateEvent()
            if self._swayControllerEnabled and self._swayController is not None:
                self._swayController.setSensorData(robot)
                self._swayController.setUpdateEvent()
            time.sleep(self._timestep)
            timestamp += self._timestep

    def launchViewer(self) -> None:
        assert mujoco is not None
        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            while viewer.is_running() and not self._shutdownEvent.is_set():
                loop_start = time.perf_counter()
                self._sendSensorData()
                self._applyControl()
                mujoco.mj_step(self._model, self._data)
                viewer.sync()
                sleep_duration = self._timestep - (time.perf_counter() - loop_start)
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

    def _sendSensorData(self) -> None:
        timestamp = float(self._data.time)
        qpos = self._data.qpos
        qvel = self._data.qvel
        drone_position = np.array(qpos[:3]) if qpos.size >= 3 else np.zeros(3)
        drone_velocity = np.array(qvel[:3]) if qvel.size >= 3 else np.zeros(3)
        drone_orientation = quaternionToEuler(qpos[3:7]) if qpos.size >= 7 else np.zeros(3)
        drone_angular_velocity = np.array(qvel[3:6]) if qvel.size >= 6 else np.zeros(3)

        sensor = SensorData(
            timestamp=timestamp,
            timestep=self._model.opt.timestep,
            dronePosition=drone_position,
            droneVelocity=drone_velocity,
            droneOrientation=drone_orientation,
            droneAngularVelocity=drone_angular_velocity,
        )
        if self._droneController is not None:
            self._droneController.setSensorData(sensor)
            self._droneController.setUpdateEvent()
        if self._telemetryBuffer is not None:
            self._telemetryBuffer.append(timestamp, drone_position, drone_orientation)

        if self._swayControllerEnabled and self._swayController is not None:
            joint_position, joint_velocity = extract_data(self._model, self._data, JOINT_NAMES)
            q_combination_case = 0
            if (
                self._droneController is not None
                and hasattr(self._droneController, "_previousControlMessage")
                and hasattr(self._droneController._previousControlMessage, "qCombinationCase")
            ):
                q_combination_case = self._droneController._previousControlMessage.qCombinationCase
            robot_sensor = RobotSensorData(
                timestamp=timestamp,
                timestep=self._model.opt.timestep,
                manipulatorPosition=drone_position + np.array([0.0, 0.0, -0.6]),
                dronePosition=drone_position,
                jointsPosition=joint_position,
                jointsVelocity=joint_velocity,
                qCombinationCase=q_combination_case,
            )
            self._swayController.setSensorData(robot_sensor)
            self._swayController.setUpdateEvent()

    def _applyControl(self) -> None:
        if self._fixedRotorThrust is not None:
            all_actions = {
                "rotor1": self._fixedRotorThrust,
                "rotor2": self._fixedRotorThrust,
                "rotor3": self._fixedRotorThrust,
                "rotor4": self._fixedRotorThrust,
                "rotor5": self._fixedRotorThrust,
                "rotor6": self._fixedRotorThrust,
            }
        elif self._droneController is not None:
            all_actions = self._getDroneControlActions()
        else:
            all_actions = {}

        if self._swayControllerEnabled and self._swayController is not None:
            all_actions.update(self._swayController.getControlInputs().jointMotors)
        for actuator_name, value in all_actions.items():
            try:
                actuator_id = self._model.actuator(actuator_name).id
            except Exception:
                continue
            self._data.ctrl[actuator_id] = value

    def stop(self) -> None:
        self._shutdownEvent.set()

    def _getDroneControlActions(self) -> dict[str, float]:
        if self._droneController is None:
            return {}
        control_input = self._droneController.getControlInput()
        u = np.asarray(control_input.u, dtype=float).reshape(-1)
        return {f"rotor{i + 1}": float(value) for i, value in enumerate(u)}

    def printBalanceDiagnostics(self) -> None:
        if mujoco is None or self._model is None or self._data is None:
            return
        mujoco.mj_forward(self._model, self._data)
        total_mass = float(np.sum(self._model.body_mass))
        rotor_count = 6
        hover_thrust_per_rotor = total_mass * 9.81 / rotor_count
        try:
            drone_body_id = self._model.body("drone").id
            drone_com = self._data.subtree_com[drone_body_id].copy()
        except Exception:
            drone_com = np.zeros(3)

        rotor_positions = []
        for i in range(1, 7):
            try:
                site_id = self._model.site(f"rotor{i}_site").id
                rotor_positions.append(self._data.site_xpos[site_id].copy())
            except Exception:
                pass
        if rotor_positions:
            rotor_center = np.mean(np.vstack(rotor_positions), axis=0)
            offset = drone_com - rotor_center
            print(f"[MujocoSimulation] Total model mass: {total_mass:.3f} kg")
            print(f"[MujocoSimulation] Hover thrust estimate: {hover_thrust_per_rotor:.3f} N per rotor")
            print(f"[MujocoSimulation] Drone subtree COM: {drone_com}")
            print(f"[MujocoSimulation] Rotor center: {rotor_center}")
            print(f"[MujocoSimulation] COM - rotor center offset: {offset}")


def quaternionToEuler(q):
    w, x, y, z = np.asarray(q, dtype=float)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=float)
