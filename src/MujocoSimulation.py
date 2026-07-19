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

from .AerialManipulator import AerialManipulator
from .Environment import Environment


class MujocoSimulation(threading.Thread):
    """MuJoCo 仿真线程（仿真整体打包）。

    主循环每帧推进 1 个物理步（1 ms），按 realTimeFactor 调速：

        sensor = Multirotor.get_state()
        -> droneController.setSensorData(sensor)   （若有控制器）
        -> TelemetryBuffer.append(...)             （若有遥测缓冲）
        -> Multirotor.set_thrusts(u)               （控制器输出；否则固定推力）
        -> TelemetryPublisher.publish(...)         （若配置 telemetryTarget，非阻塞）
        -> Environment.step()

    默认飞行控制：未注入 droneController 且未给 fixedRotorThrust 时，
    ``_build()`` 自动以出生点为目标注入 :class:`MultirotorController`
    位置闭环悬停——网格臂质心偏离悬挂轴线，开环等推力悬停物理发散
    （实测 1 s 滚到 156°），开环仅供链路验证。

    构型开关 ``withManipulator``：True（默认）为多旋翼 + 机械臂整机；
    False 为纯多旋翼平台（机械臂相关的控制器/目标会被忽略并告警）。

    跨进程输出：``telemetryTarget=(host, port)`` 指定后，每帧状态
    （位置/速度/姿态/推力，有臂时含关节角与末端位置）经
    :class:`~src.TelemetryPublisher.TelemetryPublisher` 以 UDP+JSON 发往
    另一进程；发送在独立线程完成，主循环只做一次 ``put_nowait``，零阻塞。
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
        withManipulator: bool = True,
        telemetryTarget: tuple[str, int] | None = None,
        telemetryRateHz: float = 100.0,
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
        self._withManipulator = bool(withManipulator)
        self._telemetryTarget = telemetryTarget          # (host, port)，None 关闭
        self._telemetryRateHz = max(1e-3, float(telemetryRateHz))

        self.env: Environment | None = None
        self.uam: AerialManipulator | None = None
        self._telemetryPublisher = None                  # TelemetryPublisher | None
        self._telemetryDivisor = 1                       # 抽稀因子（按物理帧计）
        self._frameIndex = 0

    # ---------- 组装 ----------

    def _build(self) -> None:
        """构建场景并把机器人挂进去统一编译，随后绑定注入的控制器。"""
        self.env = Environment(self._environmentPath)
        self.uam = AerialManipulator(compileModel=False, withManipulator=self._withManipulator)
        self.env.attach_robot(self.uam)
        # 默认飞行控制：未显式给控制器/固定推力时，注入位置闭环悬停于出生点。
        # （网格臂质心偏离悬挂轴线，开环等推力悬停物理发散，必然翻滚。）
        if self._droneController is None and self._fixedRotorThrust is None:
            from .MultirotorController import MultirotorController  # 延迟导入

            spawn = self.uam.multirotor.get_state().dronePosition
            self._droneController = MultirotorController(targetPosition=spawn)
            print(f"[MujocoSimulation] 默认注入位置闭环悬停，目标 {np.round(spawn, 3)}")
        # 两段式控制器：模型编译完成后回调 bind()，由控制器自省几何/质量
        if hasattr(self._droneController, "bind"):
            self._droneController.bind(self.uam.multirotor)
        if self._manipulatorController is not None:
            if self.uam.has_manipulator:
                self._manipulatorController.bind(self.uam.manipulator)
            else:
                print("[MujocoSimulation] 警告：纯多旋翼构型（无机械臂），忽略 manipulatorController。")
                self._manipulatorController = None
        if self._jointTargets is not None and self._manipulatorController is None:
            if self.uam.has_manipulator:
                self.uam.manipulator.set_joint_targets(self._jointTargets)
            else:
                print("[MujocoSimulation] 警告：纯多旋翼构型（无机械臂），忽略 jointTargets。")

    # ---------- 线程入口 ----------

    def run(self) -> None:
        if mujoco is None:
            print("[MujocoSimulation] mujoco 未安装，进入 clock-only 空转模式。")
            self._runWithoutMujoco()
            return

        self._build()
        self._printDiagnostics()
        self._startTelemetryChannel()

        try:
            if self._useViewer:
                self._launchViewer()
            else:
                self._runHeadless()
        finally:
            self._stopTelemetryChannel()
        print("[MujocoSimulation] 仿真结束。")

    def stop(self) -> None:
        self._shutdownEvent.set()

    # ---------- 跨进程遥测通道 ----------

    def _startTelemetryChannel(self) -> None:
        """启动 UDP 遥测发布器（独立线程，主循环只做非阻塞入队）。"""
        if self._telemetryTarget is None:
            return
        from .TelemetryPublisher import TelemetryPublisher  # 延迟导入

        assert self.env is not None
        host, port = self._telemetryTarget
        dt = float(self.env.model.opt.timestep)
        self._telemetryDivisor = max(1, int(round(1.0 / (dt * self._telemetryRateHz))))
        self._telemetryPublisher = TelemetryPublisher(host, port)
        self._telemetryPublisher.start()
        actual_hz = 1.0 / (dt * self._telemetryDivisor)
        print(f"[MujocoSimulation] 遥测通道: udp://{host}:{port} @ {actual_hz:.1f} Hz（JSON）")

    def _stopTelemetryChannel(self) -> None:
        if self._telemetryPublisher is not None:
            publisher = self._telemetryPublisher
            self._telemetryPublisher = None
            publisher.stop()
            if publisher.dropped:
                print(f"[MujocoSimulation] 遥测通道关闭：发送 {publisher.sent} 帧，丢弃 {publisher.dropped} 帧")

    def _publishTelemetry(self, sensor: SensorData, u: np.ndarray) -> None:
        """按抽稀因子向另一进程发布一帧状态（put_nowait，永不阻塞主循环）。"""
        self._frameIndex += 1
        if self._frameIndex % self._telemetryDivisor:
            return
        assert self.uam is not None
        packet = {
            "t": float(sensor.timestamp),
            "pos": sensor.dronePosition.tolist(),
            "vel": sensor.droneVelocity.tolist(),
            "rpy": sensor.droneOrientation.tolist(),
            "omega": sensor.droneAngularVelocity.tolist(),
            "thrusts": np.asarray(u, dtype=float).tolist(),
        }
        if self.uam.has_manipulator:
            packet["joints"] = self.uam.manipulator.get_joint_positions().tolist()
            packet["ee"] = self.uam.manipulator.get_ee_pose()[0].tolist()
        self._telemetryPublisher.publish(packet)

    # ---------- 主循环 ----------

    def _frame(self) -> None:
        """单帧：传感 -> 控制器/遥测 -> 控制写入 -> 物理步进。"""
        assert self.env is not None and self.uam is not None
        sensor = self.uam.multirotor.get_state()

        if self._droneController is not None:
            self._droneController.setSensorData(sensor)
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
        if self._telemetryPublisher is not None:
            self._publishTelemetry(sensor, u)
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
            time.sleep(timestep)
            timestamp += timestep

    # ---------- 诊断 ----------

    def _printDiagnostics(self) -> None:
        assert self.env is not None and self.uam is not None
        model = self.env.model
        total_mass = float(model.body_mass.sum())
        state = self.uam.multirotor.get_state()

        print(f"[MujocoSimulation] 场景 geom 数: {model.ngeom}, 障碍物: {sorted(self.env.obstacles)}")
        config = "多旋翼 + 机械臂" if self.uam.has_manipulator else "仅多旋翼"
        print(f"[MujocoSimulation] 构型: {config}")
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

