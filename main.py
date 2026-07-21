"""MuJoCo 空中机械臂仿真器 —— 程序入口。

三线程模型（Simulation 不接收 Controller 对象）：

    main.py
      ├── threading.Event        全局退出信号
      ├── TelemetryBuffer        遥测缓冲（仿真线程写，供外部读取）
      ├── MujocoSimulation       仿真线程（唯一访问 mjData，仅编译 Environment）
      │     └── wait_ready 后暴露 drone_channel / arm_channel
      ├── MultirotorController   独立控制线程（--fixed-thrust 时不启动）
      └── ManipulatorController  独立控制线程（--ee-target/--joint-targets 时启动）

接线顺序：sim.start() -> sim.wait_ready() -> 构造并 start 控制器线程
-> sim.start_physics() 放行物理。

示例：

    python main.py                          # 默认：位置闭环悬停于出生点 + viewer
    python main.py --no-viewer              # 纯无头运行
    python main.py --fixed-thrust 3.7       # 固定推力开环（非对称载荷下会翻滚，仅链路验证）
    python main.py --joint-targets 0.4 0.0  # 机械臂关节目标角
    python main.py --pos-target 0 0 2.0     # 位置闭环（MultirotorController）
    python main.py --ee-target 0.3 0 1.2    # 末端位置闭环（ManipulatorController）
    python main.py --no-arm                 # 纯多旋翼构型（不挂机械臂）
    python main.py --telemetry-out 127.0.0.1:9100   # 状态经 UDP+JSON 发往另一进程
"""

from __future__ import annotations

import argparse
import threading
import time

from src import TelemetryBuffer
from src import (
    AerialManipulator,
    Environment,
    Manipulator,
    ManipulatorController,
    MujocoSimulation,
    Multirotor,
    MultirotorController,
)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MuJoCo Aerial Manipulator Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env", "-m", type=str, default=None,
                        help="场景 MJCF 路径（默认 models/environment.xml）")
    parser.add_argument("--no-viewer", action="store_true",
                        help="不打开 MuJoCo viewer（无头运行）")
    parser.add_argument("--fixed-thrust", type=float, default=None, metavar="N",
                        help="固定旋翼推力 [N]（开环！非对称载荷下会翻滚，仅供链路验证；"
                             "缺省为位置闭环悬停）")
    parser.add_argument("--pos-target", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="位置闭环目标 [m]（启用 MultirotorController）")
    parser.add_argument("--joint-targets", type=float, nargs="+", default=None,
                        metavar="RAD",
                        help="机械臂各关节目标角 [rad]（缺省全 0）")
    parser.add_argument("--ee-target", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="机械臂末端世界系目标位置 [m]（启用 ManipulatorController）")
    parser.add_argument("--no-arm", action="store_true",
                        help="纯多旋翼构型：不挂载机械臂（忽略机械臂相关参数）")
    parser.add_argument("--telemetry-out", type=str, default=None, metavar="HOST:PORT",
                        help="把每帧状态经 UDP+JSON 发往另一进程（如 127.0.0.1:9100）")
    parser.add_argument("--telemetry-rate", type=float, default=100.0, metavar="HZ",
                        help="遥测发送频率上限 [Hz]（默认 100，按物理帧抽稀）")
    parser.add_argument("--rtf", type=float, default=1.0,
                        help="实时因子（默认 1.0）")
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()

    if args.no_arm and (args.joint_targets is not None or args.ee_target is not None):
        print("[Main] 警告：--no-arm 构型下 --joint-targets / --ee-target 将被忽略。")

    telemetry_target = None
    if args.telemetry_out is not None:
        host, sep, port_text = args.telemetry_out.rpartition(":")
        if not sep or not port_text.isdigit():
            raise SystemExit("[Main] --telemetry-out 需要 HOST:PORT 形式，例如 127.0.0.1:9100")
        telemetry_target = (host or "127.0.0.1", int(port_text))

    shutdown_event = threading.Event()
    telemetry = TelemetryBuffer()

    # ---- 组装：组件 -> 机器人 -> 场景 + 机器人 -> 仿真线程 ----
    env = Environment(args.env)
    uam = AerialManipulator(
        Multirotor(),
        None if args.no_arm else Manipulator(),
        compileModel=False,
    )
    sim = MujocoSimulation(
        shutdown_event,
        env,
        uam,
        telemetryBuffer=telemetry,
        fixedRotorThrust=args.fixed_thrust,
        useViewer=not args.no_viewer,
        realTimeFactor=args.rtf,
        telemetryTarget=telemetry_target,
        telemetryRateHz=args.telemetry_rate,
    )
    sim.start()
    if not sim.wait_ready(timeout=15.0):
        shutdown_event.set()
        sim.join(timeout=2.0)
        raise SystemExit("[Main] 仿真场景编译超时。")

    # ---- 控制器线程接线（物理停放中，构造读视图无竞争） ----
    controllers = []
    if args.fixed_thrust is None:
        # 默认：位置闭环悬停于出生点（开环等推力在非对称载荷下物理发散）
        target = args.pos_target
        if target is None:
            target = sim.uam.multirotor.get_state().dronePosition
            print(f"[Main] 默认位置闭环悬停于出生点 {target.round(3)}")
        drone_controller = MultirotorController(
            sim.drone_channel, sim.uam.multirotor, targetPosition=target
        )
        drone_controller.start()
        controllers.append(drone_controller)
    if not args.no_arm and (args.joint_targets is not None or args.ee_target is not None):
        arm_controller = ManipulatorController(
            sim.arm_channel, sim.uam.manipulator, jointTargets=args.joint_targets
        )
        if args.ee_target is not None:
            arm_controller.set_target_ee(args.ee_target)
        arm_controller.start()
        controllers.append(arm_controller)

    sim.start_physics()

    try:
        print("[Main] 无界面模式运行中，Ctrl+C 退出。")
        while sim.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Main] 收到中断信号。")
    finally:
        shutdown_event.set()
        sim.join(timeout=2.0)
        for controller in controllers:
            controller.join(timeout=1.0)


if __name__ == "__main__":
    main()
