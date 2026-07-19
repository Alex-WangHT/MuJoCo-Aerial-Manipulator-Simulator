"""MuJoCo 空中机械臂仿真器 —— 程序入口。

调用链：

    main.py
      ├── threading.Event        全局退出信号
      ├── TelemetryBuffer        遥测缓冲（仿真线程写，GCS 读）
      ├── MujocoSimulation       仿真线程（场景 + 机器人 + 控制 + viewer）
      │     ├── MultirotorController    （--pos-target 时注入）
      │     └── ManipulatorController   （--ee-target 时注入）
      └── GCS                    Tkinter 地面站（--nogui 或模块缺失时跳过）

示例：

    python main.py                          # 默认场景 + 悬停推力 + viewer + GCS
    python main.py --nogui --no-viewer      # 纯无头运行
    python main.py --fixed-thrust 3.5       # 固定推力开环
    python main.py --joint-targets 0.4 0.0  # 机械臂关节目标角
    python main.py --pos-target 0 0 2.0     # 位置闭环（MultirotorController）
    python main.py --ee-target 0.3 0 1.2    # 末端位置闭环（ManipulatorController）
"""

from __future__ import annotations

import argparse
import threading
import time

from src.MujocoSimulation import TelemetryBuffer
from src.MujocoSimulation import (
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
                        help="固定旋翼推力 [N]（缺省为整机悬停推力）")
    parser.add_argument("--pos-target", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="位置闭环目标 [m]（启用 MultirotorController）")
    parser.add_argument("--joint-targets", type=float, nargs="+", default=None,
                        metavar="RAD",
                        help="机械臂各关节目标角 [rad]（缺省全 0）")
    parser.add_argument("--ee-target", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="机械臂末端世界系目标位置 [m]（启用 ManipulatorController）")
    parser.add_argument("--rtf", type=float, default=1.0,
                        help="实时因子（默认 1.0）")
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()

    shutdown_event = threading.Event()
    telemetry = TelemetryBuffer()

    # 两段式控制器：先构造注入，仿真线程组装时自动 bind 到视图
    drone_controller = None
    if args.pos_target is not None:
        drone_controller = MultirotorController(targetPosition=args.pos_target)
    manipulator_controller = None
    if args.ee_target is not None:
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
