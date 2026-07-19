"""仿真线程。

主循环：Environment.step -> Multirotor.get_state -> droneController.setSensorData
（若有控制器）-> TelemetryBuffer.append（若有遥测）；支持被动 viewer 与无头模式，
支持时钟漂移补偿的软实时；无 mujoco 时退化为 clock-only 模式。
"""

from __future__ import annotations

import threading
import time

import numpy as np

try:
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None

from .Messages import SensorData
from .Telemetry import TelemetryBuffer

from ..AerialManipulator import AerialManipulator
from ..Environment import Environment


class MujocoSimulation(threading.Thread):
    """MuJoCo 仿真线程（仿真整体打包）。

    主循环每帧推进 1 个物理步（1 ms），按 realTimeFactor 调速：

        sensor = Multirotor.get_state()
        -> droneController.setSensorData(sensor)   （若有控制器）
        -> TelemetryBuffer.append(...)             （若有遥测）
        -> Multirotor.set_thrusts(u)               （控制器输出，否则固定/悬停推力）
        -> Environment.step()
    """

    def __init__(
        self,
        shutdownEvent: threading.Event,
        environmentPath: str | None = None,
        telemetryBuffer: TelemetryBuffer | None = None,
        droneController=None,
        manipulatorController=None,
        fixedRotorThrust: float | None = None,
        jointTargets=None,
        useViewer: bool = True,
        realTimeFactor: float = 1.0,
    ):
        super().__init__(daemon=True)
        self._shutdownEvent = shutdownEvent
        self._environmentPath = environmentPath
        self._telemetryBuffer = telemetryBuffer
        self._droneController = droneController          # 显式注入，可为 None
        self._manipulatorController = manipulatorController  # 显式注入，可为 None
        self._fixedRotorThrust = fixedRotorThrust
        self._jointTargets = (
            None if jointTargets is None else np.asarray(jointTargets, dtype=float).reshape(-1)
        )
        self._useViewer = useViewer
        self._rtf = max(1e-6, float(realTimeFactor))

        self.env: Environment | None = None
        self.uam: AerialManipulator | None = None

    # ---------- 组装 ----------

    def _build(self) -> None:
        """构建场景并把机器人挂进去统一编译，随后绑定注入的控制器。"""
        self.env = Environment(self._environmentPath)
        self.uam = AerialManipulator(compileModel=False)
        self.env.attach_robot(self.uam)
        # 两段式控制器：模型编译完成后回调 bind()，由控制器自省几何/质量
        if hasattr(self._droneController, "bind"):
            self._droneController.bind(self.uam.multirotor)
        if self._manipulatorController is not None:
            self._manipulatorController.bind(self.uam.manipulator)
        if self._jointTargets is not None and self._manipulatorController is None:
            self.uam.manipulator.set_joint_targets(self._jointTargets)

    # ---------- 线程入口 ----------

    def run(self) -> None:
        if mujoco is None:
            print("[MujocoSimulation] mujoco 未安装，进入 clock-only 空转模式。")
            self._runWithoutMujoco()
            return

        self._build()
        self._printDiagnostics()

        if self._useViewer:
            self._launchViewer()
        else:
            self._runHeadless()
        print("[MujocoSimulation] 仿真结束。")

    def stop(self) -> None:
        self._shutdownEvent.set()

    # ---------- 主循环 ----------

    def _frame(self) -> None:
        """单帧：传感 -> 控制器/遥测 -> 控制写入 -> 物理步进。"""
        assert self.env is not None and self.uam is not None
        sensor = self.uam.multirotor.get_state()

        if self._droneController is not None:
            self._droneController.setSensorData(sensor)
            self._droneController.setUpdateEvent()
        if self._telemetryBuffer is not None:
            self._telemetryBuffer.append(
                sensor.timestamp, sensor.dronePosition, sensor.droneOrientation
            )

        if self._droneController is not None:
            u = np.asarray(self._droneController.getControlInput().u, dtype=float).reshape(-1)
        else:
            # 无控制器：固定推力；缺省为整机悬停推力
            thrust = (
                self._fixedRotorThrust
                if self._fixedRotorThrust is not None
                else self.uam.hover_thrust()
            )
            u = np.full(self.uam.multirotor.n_rotors, thrust)

        self.uam.multirotor.set_thrusts(u)
        if self._manipulatorController is not None:
            self._manipulatorController.update()
        self.env.step()

    def _launchViewer(self) -> None:
        import mujoco.viewer  # 延迟导入：无显示环境下不影响无头模式

        with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
            while viewer.is_running() and not self._shutdownEvent.is_set():
                loop_start = time.perf_counter()
                self._frame()
                viewer.sync()
                self._pace(loop_start)

    def _runHeadless(self) -> None:
        while not self._shutdownEvent.is_set():
            loop_start = time.perf_counter()
            self._frame()
            self._pace(loop_start)

    def _pace(self, loop_start: float) -> None:
        """按 realTimeFactor 把每帧对齐到物理步长的墙钟时间。"""
        frame_wall = float(self.env.model.opt.timestep) / self._rtf
        remaining = frame_wall - (time.perf_counter() - loop_start)
        if remaining > 0:
            time.sleep(remaining)

    # ---------- 降级模式（无 mujoco） ----------

    def _runWithoutMujoco(self) -> None:
        timestamp = 0.0
        timestep = 0.001
        position = np.zeros(3)
        euler = np.zeros(3)
        while not self._shutdownEvent.is_set():
            sensor = SensorData(timestamp=timestamp, timestep=timestep)
            if self._telemetryBuffer is not None:
                self._telemetryBuffer.append(timestamp, position, euler)
            if self._droneController is not None:
                self._droneController.setSensorData(sensor)
                self._droneController.setUpdateEvent()
            time.sleep(timestep)
            timestamp += timestep

    # ---------- 诊断 ----------

    def _printDiagnostics(self) -> None:
        assert self.env is not None and self.uam is not None
        model = self.env.model
        total_mass = float(model.body_mass.sum())
        state = self.uam.multirotor.get_state()

        print(f"[MujocoSimulation] 场景 geom 数: {model.ngeom}, 障碍物: {sorted(self.env.obstacles)}")
        print(f"[MujocoSimulation] 整机质量: {total_mass:.3f} kg, 悬停推力: {self.uam.hover_thrust():.3f} N/rotor")
        print(f"[MujocoSimulation] 出生位置: {np.round(state.dronePosition, 3)}")

        drone_com = self.env.data.subtree_com[self.uam.multirotor.body_id].copy()
        rotor_positions = []
        for rotor_name in self.uam.multirotor.rotor_names:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, rotor_name + "_site")
            if site_id >= 0:
                rotor_positions.append(self.env.data.site_xpos[site_id].copy())
        if rotor_positions:
            rotor_center = np.mean(np.vstack(rotor_positions), axis=0)
            print(f"[MujocoSimulation] COM - 旋翼中心偏移: {np.round(drone_com - rotor_center, 4)}")

