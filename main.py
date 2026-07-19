"""MuJoCo 空中机械臂仿真器 —— 程序入口。

调用链：

    main.py
      ├── threading.Event        全局退出信号
      ├── TelemetryBuffer        遥测缓冲（仿真线程写，GCS 读）
      ├── MujocoSimulation       仿真线程（场景 + 机器人 + 控制 + viewer）
      │     ├── MultirotorController    （--pos-target 时注入）
      │     ├── ManipulatorController   （--ee-target 时注入，--no-arm 时忽略）
      │     └── TelemetryPublisher      （--telemetry-out 时启用，UDP 发往另一进程）
      └── GCS                    Tkinter 地面站（--nogui 或模块缺失时跳过）

示例：

    python main.py                          # 默认：位置闭环悬停于出生点 + viewer + GCS
    python main.py --nogui --no-viewer      # 纯无头运行
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
    ManipulatorController,
    MujocoSimulation,
    MultirotorController,
)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MuJoCo Aerial Manipulator Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env", "-m", type=str, default=None,
                        help="场景 MJCF 路径（默认 models/environment.xml）")
    parser.add_argument("--nogui", action="store_true",
                        help="不打开 GCS 地面站窗口")
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

    # 两段式控制器：先构造注入，仿真线程组装时自动 bind 到视图
    drone_controller = None
    if args.pos_target is not None:
        drone_controller = MultirotorController(targetPosition=args.pos_target)
    manipulator_controller = None
    if not args.no_arm and args.ee_target is not None:
        manipulator_controller = ManipulatorController(jointTargets=args.joint_targets)
        manipulator_controller.set_target_ee(args.ee_target)

    sim = MujocoSimulation(
        shutdown_event,
        environmentPath=args.env,
        telemetryBuffer=telemetry,
        droneController=drone_controller,
        manipulatorController=manipulator_controller,
        fixedRotorThrust=args.fixed_thrust,
        jointTargets=args.joint_targets,
        useViewer=not args.no_viewer,
        realTimeFactor=args.rtf,
        withManipulator=not args.no_arm,
        telemetryTarget=telemetry_target,
        telemetryRateHz=args.telemetry_rate,
    )
    sim.start()

    try:
        if args.nogui:
            print("[Main] 无界面模式运行中，Ctrl+C 退出。")
            while sim.is_alive():
                time.sleep(0.5)
        else:
            try:
                import tkinter as tk  # 延迟导入（GUI 依赖仅在需要时加载）

                from src.GroundControlStation import GCS
            except ImportError:
                print("[Main] 地面站模块不可用（src/GroundControlStation.py 缺失），"
                      "转为无界面模式，Ctrl+C 退出。")
                while sim.is_alive():
                    time.sleep(0.5)
            else:
                root = tk.Tk()
                GCS(root, taskManager=None, shutdownEvent=shutdown_event,
                    telemetryBuffer=telemetry)
                root.mainloop()
    except KeyboardInterrupt:
        print("\n[Main] 收到中断信号。")
    finally:
        shutdown_event.set()
        sim.join(timeout=2.0)


if __name__ == "__main__":
    main()
