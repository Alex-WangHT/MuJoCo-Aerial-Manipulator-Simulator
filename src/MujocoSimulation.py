"""仿真线程。

三线程模型：MujocoSimulation（物理，唯一访问 mjData）+
MultirotorController / ManipulatorController（各自独立线程）。
Simulation **不接收 Controller 对象**——它在编译后创建并暴露两条
帧同步通道（``drone_channel`` / ``arm_channel``），控制器由外部
（如 main.py）在 ``wait_ready()`` 之后构造到通道上并 ``start()``，
最后 ``start_physics()`` 放行物理推进。

主循环每帧：发布传感快照 -> 等控制器回执 -> 帧边界 flush 执行器缓冲
-> Environment.step()；无订阅者时退化为固定推力开环。
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
from .FrameSync import ControllerChannel


class MujocoSimulation(threading.Thread):
    """MuJoCo 仿真线程（仿真整体打包，仅对 Environment 编译）。

    **输入为合并好的两大实例**：外部构造好的 ``Environment`` 与未编译的
    ``AerialManipulator``（``compileModel=False``）；本线程把机器人挂进
    场景统一编译（``Environment.attach_robot``），即"Environment 和
    AerialManipulator 合并打包进 MujocoSimulation"。

    主循环每帧推进 1 个物理步（1 ms），按 realTimeFactor 调速：

        有控制器订阅时（帧同步协议，控制频率 = 物理帧率）：
            drone_channel: publish(SensorData) -> wait_done -> flush 旋翼缓冲
            arm_channel:   publish(ManipulatorSensorData) -> wait_done -> flush 舵机缓冲
        无订阅者时（开环）：固定推力（缺省整机悬停推力）
        -> TelemetryBuffer.append / TelemetryPublisher.publish（若配置）
        -> Environment.step()

    三线程接线（Simulation 不接收 Controller 对象）::

        env = Environment()
        uam = AerialManipulator(Multirotor(), Manipulator(), compileModel=False)
        sim = MujocoSimulation(shutdown, env, uam, useViewer=False)
        sim.start()
        sim.wait_ready()                       # 场景编译完成，通道已暴露
        drone = MultirotorController(sim.drone_channel, sim.uam.multirotor, ...)
        arm = ManipulatorController(sim.arm_channel, sim.uam.manipulator, ...)
        drone.start(); arm.start()             # 两个控制器各自独立线程
        sim.start_physics()                    # 放行物理推进

    ``wait_ready()`` 到 ``start_physics()`` 之间物理处于停放状态，
    控制器构造（读视图初始状态）无数据竞争。

    纯多旋翼构型：传入未带机械臂的 uam（``AerialManipulator(Multirotor())``）
    即可，``arm_channel`` 自动为 ``None``。

    跨进程输出：``telemetryTarget=(host, port)`` 指定后，每帧状态经
    :class:`~src.TelemetryPublisher.TelemetryPublisher` 以 UDP+JSON 发往
    另一进程；发送在独立线程完成，主循环只做一次 ``put_nowait``，零阻塞。
    """

    def __init__(
        self,
        shutdownEvent: threading.Event,
        environment: Environment,
        uam: AerialManipulator,
        telemetryBuffer: TelemetryBuffer | None = None,
        fixedRotorThrust: float | None = None,
        useViewer: bool = True,
        realTimeFactor: float = 1.0,
        telemetryTarget: tuple[str, int] | None = None,
        telemetryRateHz: float = 100.0,
    ):
        super().__init__(daemon=True)
        if not isinstance(environment, Environment):
            raise TypeError(
                f"MujocoSimulation: environment 应为 Environment 实例，收到 {type(environment).__name__}"
            )
        if not isinstance(uam, AerialManipulator):
            raise TypeError(
                f"MujocoSimulation: uam 应为 AerialManipulator 实例（compileModel=False），"
                f"收到 {type(uam).__name__}"
            )
        self._shutdownEvent = shutdownEvent
        self._telemetryBuffer = telemetryBuffer
        self._fixedRotorThrust = fixedRotorThrust
        self._useViewer = useViewer
        self._rtf = max(1e-6, float(realTimeFactor))
        self._telemetryTarget = telemetryTarget          # (host, port)，None 关闭
        self._telemetryRateHz = max(1e-3, float(telemetryRateHz))

        self.env: Environment | None = environment       # _build 中完成合并
        self.uam: AerialManipulator | None = uam
        self.drone_channel: ControllerChannel | None = None
        self.arm_channel: ControllerChannel | None = None   # 纯多旋翼构型为 None
        self._telemetryPublisher = None                  # TelemetryPublisher | None
        self._telemetryDivisor = 1                       # 抽稀因子（按物理帧计）
        self._frameIndex = 0
        self._ready = threading.Event()                  # 编译完成、通道已暴露
        self._gate = threading.Event()                   # start_physics() 放行

    # ---------- 对外握手 ----------

    def wait_ready(self, timeout: float | None = None) -> bool:
        """等待场景编译完成（通道已暴露、物理仍停放），返回是否就绪。"""
        return self._ready.wait(timeout)

    def start_physics(self) -> None:
        """放行物理推进（控制器线程应在此之前构造并 start）。"""
        self._gate.set()

    # ---------- 组装 ----------

    def _build(self) -> None:
        """把外部传入的机器人合并进场景统一编译，随后创建帧同步通道。

        本线程**只对 Environment 编译**：机器人须以 ``compileModel=False``
        构建，唯一的编译入口是 ``Environment.attach_robot()`` ->
        ``Environment.compile()``。
        """
        # Environment + AerialManipulator 合并打包（attach 内部会校验未编译）
        self.env.attach_robot(self.uam)

        # 帧同步通道：控制器（独立线程）由外部在 wait_ready 后注册
        self.drone_channel = ControllerChannel()
        self.arm_channel = ControllerChannel() if self.uam.has_manipulator else None

    # ---------- 线程入口 ----------

    def run(self) -> None:
        if mujoco is None:
            print("[MujocoSimulation] mujoco 未安装，进入 clock-only 空转模式。")
            self._ready.set()
            self._runWithoutMujoco()
            return

        self._build()
        self._printDiagnostics()
        self._startTelemetryChannel()
        self._ready.set()

        # 物理停放：等待外部接线（控制器构造 + start）完成后放行
        while not self._gate.is_set() and not self._shutdownEvent.is_set():
            self._gate.wait(0.05)

        try:
            if self._useViewer:
                self._launchViewer()
            else:
                self._runHeadless()
        finally:
            # 关闭通道：控制器线程的 wait_snapshot 收到 None 后自行退出
            if self.drone_channel is not None:
                self.drone_channel.close()
            if self.arm_channel is not None:
                self.arm_channel.close()
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
        """单帧：快照发布 -> 控制器回执 -> 帧边界 flush -> 物理步进。"""
        assert self.env is not None and self.uam is not None
        sensor = self.uam.multirotor.get_state()

        if self._telemetryBuffer is not None:
            self._telemetryBuffer.append(
                sensor.timestamp, sensor.dronePosition, sensor.droneOrientation
            )

        # ---- 多旋翼通道：帧同步 or 开环固定推力 ----
        drone_ch = self.drone_channel
        if drone_ch is not None and drone_ch.has_subscriber:
            drone_ch.mailbox.publish(sensor)
            drone_ch.mailbox.wait_done(self._shutdownEvent)
            drone_ch.flush()
        else:
            thrust = (
                self._fixedRotorThrust
                if self._fixedRotorThrust is not None
                else self.uam.hover_thrust()
            )
            self.uam.multirotor.set_actuator(np.full(self.uam.multirotor.n_rotors, thrust))

        # ---- 机械臂通道：帧同步（无订阅者时舵机保持 ctrl 中已有目标角） ----
        arm_ch = self.arm_channel
        if arm_ch is not None and arm_ch.has_subscriber:
            arm_snap = self.uam.manipulator.get_state()
            arm_ch.mailbox.publish(arm_snap)
            arm_ch.mailbox.wait_done(self._shutdownEvent)
            arm_ch.flush()

        if self._telemetryPublisher is not None:
            u = np.array([
                self.env.data.ctrl[self.uam.multirotor.rotors[name]]
                for name in self.uam.multirotor.rotor_names
            ])
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
        if self._fixedRotorThrust is not None:
            print(f"[MujocoSimulation] 开环固定推力 {self._fixedRotorThrust} N/rotor（非对称载荷下会翻滚）")

        drone_com = self.env.data.subtree_com[self.uam.multirotor.body_id].copy()
        rotor_positions = []
        for rotor_name in self.uam.multirotor.rotor_names:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, rotor_name + "_site")
            if site_id >= 0:
                rotor_positions.append(self.env.data.site_xpos[site_id].copy())
        if rotor_positions:
            rotor_center = np.mean(np.vstack(rotor_positions), axis=0)
            print(f"[MujocoSimulation] COM - 旋翼中心偏移: {np.round(drone_com - rotor_center, 4)}")
