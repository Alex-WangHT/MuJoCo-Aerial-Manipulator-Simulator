from __future__ import annotations

import math
import pathlib
import threading
import time

import numpy as np

from include.Messages import SensorData
from include.Telemetry import TelemetryBuffer
from src.Robot import Robot

try:
    import mujoco
    import mujoco.viewer
except Exception:  # pragma: no cover - allows framework use before installing mujoco.
    mujoco = None


class MujocoSimulation(threading.Thread):
    """MuJoCo 仿真线程。

    职责：
    - 加载 MJCF 模型并启动 passive viewer
    - 每帧从 ``Robot`` 提取状态，推给 ``DroneControllerInterface``
    - 从控制器读取控制输出，通过 ``Robot`` 写入 MuJoCo actuator
    - 向 ``TelemetryBuffer`` 写入遥测数据
    - 调用 ``mujoco.mj_step()`` 推进仿真
    """

    def __init__(
        self,
        shutdownEvent: threading.Event,
        modelPath: str | None = None,
        timestep: float = 0.001,
        telemetryBuffer: TelemetryBuffer | None = None,
        fixedRotorThrust: float | None = None,
    ):
        super().__init__(daemon=True)
        self._shutdownEvent = shutdownEvent
        self._timestep = timestep
        self._modelPath = pathlib.Path(modelPath or "models/common_uam.xml")
        self._telemetryBuffer = telemetryBuffer
        self._fixedRotorThrust = fixedRotorThrust
        self._model = None
        self._data = None
        self._robot: Robot | None = None

    def run(self):
        print("[MujocoSimulation] Initialized.")
        if mujoco is None:
            print(
                "[MujocoSimulation] mujoco package not installed; "
                "running clock-only sensor loop."
            )
            self._runWithoutMujoco()
            return

        self._model = mujoco.MjModel.from_xml_path(str(self._modelPath))
        self._data = mujoco.MjData(self._model)
        self._robot = Robot(self._model, self._data)
        self._printDiagnostics()
        self._launchViewer()
        print("[MujocoSimulation] Ending simulation.")

    # ---------- fallback loop (no mujoco) ----------

    def _runWithoutMujoco(self) -> None:
        timestamp = 0.0
        position = np.zeros(3)
        velocity = np.zeros(3)
        while not self._shutdownEvent.is_set():
            sensor = SensorData(
                timestamp=timestamp,
                timestep=self._timestep,
                dronePosition=position,
                droneVelocity=velocity,
            )
            if self._telemetryBuffer is not None:
                self._telemetryBuffer.append(timestamp, position, np.zeros(3))
            if self._droneController is not None:
                self._droneController.setSensorData(sensor)
                self._droneController.setUpdateEvent()
            time.sleep(self._timestep)
            timestamp += self._timestep

    # ---------- main sim loop ----------

    def _launchViewer(self) -> None:
        assert mujoco is not None
        assert self._robot is not None
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
        assert self._robot is not None
        sensor = self._robot.getSensorData()

        if self._droneController is not None:
            self._droneController.setSensorData(sensor)
            self._droneController.setUpdateEvent()
        if self._telemetryBuffer is not None:
            self._telemetryBuffer.append(
                sensor.timestamp, sensor.dronePosition, sensor.droneOrientation
            )

    def _applyControl(self) -> None:
        assert self._robot is not None
        actions: dict[str, float] = {}

        if self._fixedRotorThrust is not None:
            # 按 rotor1..rotor6 硬编码映射
            for i in range(1, 7):
                actions[f"rotor{i}"] = self._fixedRotorThrust
        elif self._droneController is not None:
            control_input = self._droneController.getControlInput()
            u = np.asarray(control_input.u, dtype=float).reshape(-1)
            for i, value in enumerate(u):
                actions[f"rotor{i + 1}"] = float(value)

        self._robot.applyControl(actions)

    def stop(self) -> None:
        self._shutdownEvent.set()

    # ---------- diagnostics ----------

    def _printDiagnostics(self) -> None:
        if mujoco is None or self._model is None or self._data is None:
            return
        mujoco.mj_forward(self._model, self._data)
        total_mass = float(np.sum(self._model.body_mass))
        rotor_count = 6
        hover_thrust = total_mass * 9.81 / rotor_count
        try:
            drone_body_id = self._model.body("drone").id
            drone_com = self._data.subtree_com[drone_body_id].copy()
        except Exception:
            drone_com = np.zeros(3)

        rotor_positions = []
        for i in range(1, rotor_count + 1):
            try:
                site_id = self._model.site(f"rotor{i}_site").id
                rotor_positions.append(self._data.site_xpos[site_id].copy())
            except Exception:
                pass

        print(f"[MujocoSimulation] Total model mass: {total_mass:.3f} kg")
        print(f"[MujocoSimulation] Hover thrust estimate: {hover_thrust:.3f} N per rotor")
        print(f"[MujocoSimulation] Drone subtree COM: {drone_com}")
        if rotor_positions:
            rotor_center = np.mean(np.vstack(rotor_positions), axis=0)
            offset = drone_com - rotor_center
            print(f"[MujocoSimulation] Rotor center: {rotor_center}")
            print(f"[MujocoSimulation] COM - rotor center offset: {offset}")
