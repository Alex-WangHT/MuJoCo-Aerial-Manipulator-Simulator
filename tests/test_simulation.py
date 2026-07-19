"""MujocoSimulation 整体打包的无头仿真测试。

直接运行：
    .venv/Scripts/python tests/test_simulation.py

验证内容：
1. 默认（悬停推力）无头运行 1 s：遥测按 ~1 kHz 记录，高度保持 1.5 m；
2. 固定推力 4.0 N（> 悬停值）：无人机应上升；
3. 固定推力 0：无人机应下落（并由机械臂触地支撑）；
4. stop() 后线程正常退出。
"""

from __future__ import annotations

import pathlib
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src.MujocoSimulation import MujocoSimulation, TelemetryBuffer


def run_sim(seconds: float, **kwargs) -> tuple[MujocoSimulation, TelemetryBuffer]:
    shutdown = threading.Event()
    telemetry = TelemetryBuffer()
    sim = MujocoSimulation(shutdown, telemetryBuffer=telemetry, useViewer=False, **kwargs)
    sim.start()
    time.sleep(seconds)
    shutdown.set()
    sim.join(timeout=3.0)
    return sim, telemetry


def main() -> None:
    print("=" * 64)
    print("MujocoSimulation 无头仿真测试")
    print("=" * 64)

    # ---------- 1. 默认悬停 1 s ----------
    print("\n[1] 默认悬停推力，无头运行 1 s")
    sim, telemetry = run_sim(1.0)
    assert not sim.is_alive(), "stop 后线程应退出"
    assert sim.env is not None and sim.uam is not None, "场景与机器人应已构建"
    time_arr, positions, eulers = telemetry.snapshot()
    assert len(time_arr) > 400, f"1 s 应记录约 1000 帧遥测，实际 {len(time_arr)}"
    final_z = float(positions[-1][2])
    final_rpy_deg = np.rad2deg(eulers[-1])
    print(f"    遥测帧数: {len(time_arr)}（时长 {time_arr[-1]:.3f} s）")
    print(f"    末端高度: {final_z:.4f} m, 姿态[deg]: {np.round(final_rpy_deg, 2)}")
    print(f"    障碍物: {sorted(sim.env.obstacles)}")
    assert abs(final_z - 1.5) < 0.1, f"悬停高度应保持 1.5 m，实际 {final_z:.3f}"
    assert np.max(np.abs(final_rpy_deg[:2])) < 3.0, "姿态应近水平"

    # ---------- 2. 固定推力 4.0 N -> 上升 ----------
    print("\n[2] 固定推力 4.0 N（> 悬停 3.17 N），0.8 s")
    _, telemetry_up = run_sim(0.8, fixedRotorThrust=4.0)
    _, pos_up, _ = telemetry_up.snapshot()
    print(f"    末端高度: {pos_up[-1][2]:.4f} m（初始 1.5 m）")
    assert pos_up[-1][2] > 1.6, f"推力大于悬停值应上升，实际 {pos_up[-1][2]:.3f}"

    # ---------- 3. 固定推力 0 -> 下落 ----------
    print("\n[3] 固定推力 0，0.8 s")
    _, telemetry_down = run_sim(0.8, fixedRotorThrust=0.0)
    _, pos_down, _ = telemetry_down.snapshot()
    print(f"    末端高度: {pos_down[-1][2]:.4f} m（初始 1.5 m）")
    assert pos_down[-1][2] < 1.0, f"零推力应下落，实际 {pos_down[-1][2]:.3f}"

    print("\n" + "=" * 64)
    print("PASS: 悬停遥测、固定推力上升/下落、线程退出 全部通过")
    print("=" * 64)


if __name__ == "__main__":
    main()
