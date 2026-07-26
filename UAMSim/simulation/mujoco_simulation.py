"""MuJoCo 仿真线程。

输入 Environment + 机器人组件实例（合并打包，仅对 Environment 编译），
唯一访问 mjData；编译后暴露 drone_channel / arm_channel 与 sim.uam 句柄。
"""

from __future__ import annotations

import threading
import time

import numpy as np

try:
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None

from ..utils.perception_bus import SensorSnapshot
from ..utils.telemetry import TelemetryBuffer

from .environment import Environment
from ..utils.frame_sync import ControllerChannel
from .manipulator import Manipulator
from .multirotor import Multirotor
from .robot import Robot


class MujocoSimulation(threading.Thread):
    def __init__(
        self,
        shutdown_event: threading.Event,
        environment: Environment,
        multirotor: Multirotor,
        manipulator: Manipulator | None = None,
        telemetry_buffer: TelemetryBuffer | None = None,
        fixed_rotor_thrust: float | None = None,
        use_viewer: bool = True,
        real_time_factor: float = 1.0,
        telemetry_target: tuple[str, int] | None = None,
        telemetry_rate_hz: float = 100.0,
        perception_target: tuple[str, int] | None = None,
    ):
        super().__init__(daemon=True)
        if not isinstance(environment, Environment):
            raise TypeError(
                f"MujocoSimulation: environment 应为 Environment 实例，收到 {type(environment).__name__}"
            )
        if not isinstance(multirotor, Multirotor):
            raise TypeError(
                f"MujocoSimulation: multirotor 应为 Multirotor 实例，收到 {type(multirotor).__name__}"
            )
        if manipulator is not None and not isinstance(manipulator, Manipulator):
            raise TypeError(
                f"MujocoSimulation: manipulator 应为 Manipulator 实例或 None，"
                f"收到 {type(manipulator).__name__}"
            )
        self._shutdown_event = shutdown_event
        self._multirotor = multirotor
        self._manipulator = manipulator
        self._telemetry_buffer = telemetry_buffer
        self._fixed_rotor_thrust = fixed_rotor_thrust
        self._use_viewer = use_viewer
        self._rtf = max(1e-6, float(real_time_factor))
        self._telemetry_target = telemetry_target          # (host, port)，None 关闭
        self._telemetry_rate_hz = max(1e-3, float(telemetry_rate_hz))
        self._perception_target = perception_target        # (host, port)，None 关闭
        self._perception_sources: list = []               # PerceptionSource 列表
        self._perception_publisher = None                 # PerceptionPublisher | None
        self._perception_divisors: list[int] = []         # 各源抽稀因子（按物理帧计）

        self.env: Environment | None = environment       # _build 中完成合并
        self.uam: Robot | None = None                    # _build 中 attach 后获得
        self.drone_channel: ControllerChannel | None = None
        self.arm_channel: ControllerChannel | None = None   # 纯多旋翼构型为 None
        self._telemetry_publisher = None                  # TelemetryPublisher | None
        self._telemetry_divisor = 1                       # 抽稀因子（按物理帧计）
        self._frame_index = 0
        self._ready = threading.Event()                  # 编译完成、通道已暴露
        self._gate = threading.Event()                   # start_physics() 放行

    # ---------- 对外握手 ----------

    def wait_ready(self, timeout: float | None = None) -> bool:
        """等待场景编译完成（通道已暴露、物理仍停放），返回是否就绪。"""
        return self._ready.wait(timeout)

    def start_physics(self) -> None:
        """放行物理推进（控制器线程应在此之前构造并 start）。"""
        self._gate.set()

    def add_perception_source(self, source) -> None:
        """注册感知源（相机/雷达预留接口，物理启动前调用）。

        ``source`` 为 :class:`~UAMSim.utils.perception_bus.PerceptionSource` 实例；
        仿真线程按 ``source.rate_hz`` 折算的抽稀因子在帧边界调用其
        ``capture(model, data)``，产出经 UDP 感知通道发往对端进程。
        需构造时传入 ``perception_target``，否则源注册后不会发送。
        """
        if not getattr(source, "name", None) or not callable(getattr(source, "capture", None)):
            raise TypeError("add_perception_source: source 应为 PerceptionSource 实例")
        self._perception_sources.append(source)

    # ---------- 组装 ----------

    def _build(self) -> None:
        """把外部传入的机器人合并进场景统一编译，随后创建帧同步通道。

        本线程**只对 Environment 编译**：机器人须以 ``compileModel=False``
        构建，唯一的编译入口是 ``Environment.attach_robot()`` ->
        ``Environment.compile()``。
        """
        # Environment + AerialManipulator 合并打包（attach 内部会校验未编译）
        self.uam = self.env.attach_robot(self._multirotor, self._manipulator)

        # 帧同步通道：控制器（独立线程）由外部在 wait_ready 后注册
        self.drone_channel = ControllerChannel()
        self.arm_channel = ControllerChannel() if self.uam.has_manipulator else None

    # ---------- 线程入口 ----------

    def run(self) -> None:
        if mujoco is None:
            print("[MujocoSimulation] mujoco 未安装，进入 clock-only 空转模式。")
            self._ready.set()
            self._run_without_mujoco()
            return

        self._build()
        self._print_diagnostics()
        self._start_telemetry_channel()
        self._start_perception_channel()
        self._ready.set()

        # 物理停放：等待外部接线（控制器构造 + start）完成后放行
        while not self._gate.is_set() and not self._shutdown_event.is_set():
            self._gate.wait(0.05)

        try:
            if self._use_viewer:
                self._launch_viewer()
            else:
                self._run_headless()
        finally:
            # 关闭通道：控制器线程的 wait_snapshot 收到 None 后自行退出
            if self.drone_channel is not None:
                self.drone_channel.close()
            if self.arm_channel is not None:
                self.arm_channel.close()
            self._stop_telemetry_channel()
            self._stop_perception_channel()
        print("[MujocoSimulation] 仿真结束。")

    def stop(self) -> None:
        self._shutdown_event.set()

    # ---------- 跨进程遥测通道 ----------

    def _start_telemetry_channel(self) -> None:
        """启动 UDP 遥测发布器（独立线程，主循环只做非阻塞入队）。"""
        if self._telemetry_target is None:
            return
        from ..utils.telemetry_publisher import TelemetryPublisher  # 延迟导入

        assert self.env is not None
        host, port = self._telemetry_target
        dt = float(self.env.model.opt.timestep)
        self._telemetry_divisor = max(1, int(round(1.0 / (dt * self._telemetry_rate_hz))))
        self._telemetry_publisher = TelemetryPublisher(host, port)
        self._telemetry_publisher.start()
        actual_hz = 1.0 / (dt * self._telemetry_divisor)
        print(f"[MujocoSimulation] 遥测通道: udp://{host}:{port} @ {actual_hz:.1f} Hz（JSON）")

    def _stop_telemetry_channel(self) -> None:
        if self._telemetry_publisher is not None:
            publisher = self._telemetry_publisher
            self._telemetry_publisher = None
            publisher.stop()
            if publisher.dropped:
                print(f"[MujocoSimulation] 遥测通道关闭：发送 {publisher.sent} 帧，丢弃 {publisher.dropped} 帧")

    # ---------- 跨进程感知通道（相机/雷达预留） ----------

    def _start_perception_channel(self) -> None:
        """启动 UDP 感知发布器并折算各源抽稀因子（独立线程，主循环零阻塞）。"""
        if self._perception_target is None:
            if self._perception_sources:
                print("[MujocoSimulation] 已注册感知源但未配置 perception_target，感知数据不发送。")
            return
        if not self._perception_sources:
            return
        from ..utils.perception_bus import PerceptionPublisher  # 延迟导入

        assert self.env is not None
        host, port = self._perception_target
        dt = float(self.env.model.opt.timestep)
        self._perception_publisher = PerceptionPublisher(host, port)
        self._perception_publisher.start()
        self._perception_divisors = [
            max(1, int(round(1.0 / (dt * max(1e-3, float(src.rate_hz))))))
            for src in self._perception_sources
        ]
        rates = ", ".join(
            f"{src.name}@{1.0 / (dt * div):.1f}Hz"
            for src, div in zip(self._perception_sources, self._perception_divisors)
        )
        print(f"[MujocoSimulation] 感知通道: udp://{host}:{port}（{rates}）")

    def _stop_perception_channel(self) -> None:
        if self._perception_publisher is not None:
            publisher = self._perception_publisher
            self._perception_publisher = None
            publisher.stop()
            if publisher.dropped_chunks:
                print(f"[MujocoSimulation] 感知通道关闭：发送 {publisher.sent_frames} 帧，"
                      f"丢弃 {publisher.dropped_chunks} 片")

    def _publish_perception(self) -> None:
        """按各源抽稀因子采集一帧感知数据并入队（put_nowait，永不阻塞主循环）。

        ``capture`` 在仿真线程内执行（可访问 mjData）；源的异常只打印警告，
        不影响物理推进。
        """
        assert self.env is not None
        for src, div in zip(self._perception_sources, self._perception_divisors):
            if self._frame_index % div:
                continue
            try:
                payload, meta = src.capture(self.env.model, self.env.data)
            except Exception as exc:  # noqa: BLE001 - 感知源故障不应击落物理循环
                print(f"[MujocoSimulation] 感知源 '{src.name}' capture 异常: {exc}")
                continue
            self._perception_publisher.publish(src.name, payload, meta)

    def _publish_telemetry(self, sensor: SensorSnapshot, u: np.ndarray) -> None:
        """按抽稀因子向另一进程发布一帧状态（put_nowait，永不阻塞主循环）。"""
        if self._frame_index % self._telemetry_divisor:
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
        self._telemetry_publisher.publish(packet)

    # ---------- 主循环 ----------

    def _frame(self) -> None:
        """单帧：快照发布 -> 控制器回执 -> 帧边界 flush -> 物理步进。"""
        assert self.env is not None and self.uam is not None
        self._frame_index += 1
        sensor = self.uam.multirotor.get_state()

        if self._telemetry_buffer is not None:
            self._telemetry_buffer.append(
                sensor.timestamp, sensor.dronePosition, sensor.droneOrientation
            )

        # ---- 多旋翼通道：帧同步 or 开环固定推力 ----
        drone_ch = self.drone_channel
        if drone_ch is not None and drone_ch.has_subscriber:
            drone_ch.mailbox.publish(sensor)
            drone_ch.mailbox.wait_done(self._shutdown_event)
            drone_ch.flush()
        else:
            thrust = (
                self._fixed_rotor_thrust
                if self._fixed_rotor_thrust is not None
                else self.uam.hover_thrust()
            )
            self.uam.multirotor.set_actuator(np.full(self.uam.multirotor.n_rotors, thrust))

        # ---- 机械臂通道：帧同步（无订阅者时舵机保持 ctrl 中已有目标角） ----
        arm_ch = self.arm_channel
        if arm_ch is not None and arm_ch.has_subscriber:
            arm_snap = self.uam.manipulator.get_state()
            arm_ch.mailbox.publish(arm_snap)
            arm_ch.mailbox.wait_done(self._shutdown_event)
            arm_ch.flush()

        if self._telemetry_publisher is not None:
            u = np.array([
                self.env.data.ctrl[self.uam.multirotor.rotors[name]]
                for name in self.uam.multirotor.rotor_names
            ])
            self._publish_telemetry(sensor, u)
        if self._perception_publisher is not None:
            self._publish_perception()
        self.env.step()

    def _launch_viewer(self) -> None:
        import mujoco.viewer  # 延迟导入：无显示环境下不影响无头模式

        with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
            while viewer.is_running() and not self._shutdown_event.is_set():
                loop_start = time.perf_counter()
                self._frame()
                viewer.sync()
                self._pace(loop_start)

    def _run_headless(self) -> None:
        while not self._shutdown_event.is_set():
            loop_start = time.perf_counter()
            self._frame()
            self._pace(loop_start)

    def _pace(self, loop_start: float) -> None:
        """按 real_time_factor 把每帧对齐到物理步长的墙钟时间。

        自旋等待（不用 sleep）：Windows 下次毫秒级 sleep 实际会睡 1ms 以上，
        1 kHz 物理帧（预算 1ms）会被拖到 ~0.6x 实时；自旋有微秒级精度，
        代价是配速期间占满一个核——实时仿真本就该独占一个核。
        """
        deadline = loop_start + float(self.env.model.opt.timestep) / self._rtf
        while time.perf_counter() < deadline:
            pass

    # ---------- 降级模式（无 mujoco） ----------

    def _run_without_mujoco(self) -> None:
        timestamp = 0.0
        timestep = 0.001
        position = np.zeros(3)
        euler = np.zeros(3)
        while not self._shutdown_event.is_set():
            sensor = SensorSnapshot(timestamp=timestamp, timestep=timestep)
            if self._telemetry_buffer is not None:
                self._telemetry_buffer.append(timestamp, position, euler)
            time.sleep(timestep)
            timestamp += timestep

    # ---------- 诊断 ----------

    def _print_diagnostics(self) -> None:
        assert self.env is not None and self.uam is not None
        model = self.env.model
        total_mass = float(model.body_mass.sum())
        state = self.uam.multirotor.get_state()

        print(f"[MujocoSimulation] 场景 geom 数: {model.ngeom}, 障碍物: {sorted(self.env.obstacles)}")
        config = "多旋翼 + 机械臂" if self.uam.has_manipulator else "仅多旋翼"
        print(f"[MujocoSimulation] 构型: {config}")
        print(f"[MujocoSimulation] 整机质量: {total_mass:.3f} kg, 悬停推力: {self.uam.hover_thrust():.3f} N/rotor")
        print(f"[MujocoSimulation] 出生位置: {np.round(state.dronePosition, 3)}")
        if self._fixed_rotor_thrust is not None:
            print(f"[MujocoSimulation] 开环固定推力 {self._fixed_rotor_thrust} N/rotor（非对称载荷下会翻滚）")

        drone_com = self.env.data.subtree_com[self.uam.multirotor.body_id].copy()
        rotor_positions = []
        for rotor_name in self.uam.multirotor.rotor_names:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, rotor_name + "_site")
            if site_id >= 0:
                rotor_positions.append(self.env.data.site_xpos[site_id].copy())
        if rotor_positions:
            rotor_center = np.mean(np.vstack(rotor_positions), axis=0)
            print(f"[MujocoSimulation] COM - 旋翼中心偏移: {np.round(drone_com - rotor_center, 4)}")
