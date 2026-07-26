"""MujocoSimulation 三线程模型的无头仿真测试。

安装包后运行::

    python tests/test_simulation.py

验证内容：
1. 闭环悬停无头运行 1 s：控制器独立线程经 sim.drone_channel 接线，
   遥测按 ~1 kHz 记录，高度保持 1.5 m；
2. 位置闭环上升到 2.0 m；
3. 固定推力 0（无控制器线程）：无人机应下落；
4. stop() 后仿真线程与控制器线程均正常退出。
"""

from __future__ import annotations

import pathlib
import threading
import time

import numpy as np

import UAMSim as ua


_BASE = pathlib.Path(__file__).resolve().parent.parent
_SCENE_XML = str(_BASE / "examples" / "models" / "environment.xml")
_MULTIROTOR_XML = str(_BASE / "examples" / "models" / "multirotor.xml")


class HoverController(ua.MultirotorController):
    """最小 PD 高度保持控制器（测试用）。"""

    def controller(self, feedback, targetPosition, hoverThrust, kp, kd):
        err = targetPosition - feedback.dronePosition
        vel = feedback.droneVelocity
        thrust = hoverThrust + kp * err[2] - kd * vel[2]
        return np.full(self._multirotor.n_rotors, thrust)


def run_sim(seconds: float, drone_target=None, **kwargs):
    """组装场景+机器人组件并启动仿真线程；给 drone_target 时接线一个位置闭环控制器线程。"""
    shutdown = threading.Event()
    telemetry = ua.TelemetryBuffer()
    env = ua.Environment(_SCENE_XML)
    sim = ua.MujocoSimulation(
        shutdown, env, ua.Multirotor(_MULTIROTOR_XML), None,
        telemetry_buffer=telemetry, use_viewer=False, **kwargs
    )
    sim.start()
    assert sim.wait_ready(timeout=10.0), "场景编译超时"
    drone_ctrl = None
    if drone_target is not None:
        drone_ctrl = HoverController(
            sim.uam.multirotor, sim.drone_channel,
            targetPosition=drone_target,
            hoverThrust=sim.uam.hover_thrust(),
            kp=4.0, kd=3.0,
        )
        drone_ctrl.start()
    sim.start_physics()
    time.sleep(seconds)
    shutdown.set()
    sim.join(timeout=3.0)
    if drone_ctrl is not None:
        drone_ctrl.join(timeout=2.0)
        assert not drone_ctrl.is_alive(), "控制器线程应随通道关闭而退出"
    return sim, telemetry


def main() -> None:
    print("=" * 64)
    print("MujocoSimulation 无头仿真测试（三线程模型）")
    print("=" * 64)

    # ---------- 1. 闭环悬停 1 s ----------
    print("\n[1] 位置闭环悬停（目标 z=1.5 m），无头运行 1 s")
    sim, telemetry = run_sim(1.0, drone_target=[0.0, 0.0, 1.5])
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

    # ---------- 2. 位置闭环上升到 2.0 m ----------
    print("\n[2] 位置闭环上升（目标 z=2.0 m），1.5 s")
    _, telemetry_up = run_sim(1.5, drone_target=[0.0, 0.0, 2.0])
    _, pos_up, _ = telemetry_up.snapshot()
    print(f"    末端高度: {pos_up[-1][2]:.4f} m（初始 1.5 m）")
    assert pos_up[-1][2] > 1.6, f"闭环爬升应到达 2.0 m 附近，实际 {pos_up[-1][2]:.3f}"

    # ---------- 3. 固定推力 0（无控制器线程）-> 下落 ----------
    print("\n[3] 固定推力 0，0.8 s")
    _, telemetry_down = run_sim(0.8, fixed_rotor_thrust=0.0)
    _, pos_down, _ = telemetry_down.snapshot()
    print(f"    末端高度: {pos_down[-1][2]:.4f} m（初始 1.5 m）")
    assert pos_down[-1][2] < 1.0, f"零推力应下落，实际 {pos_down[-1][2]:.3f}"

    print("\n" + "=" * 64)
    print("PASS: 悬停遥测、闭环爬升、开环下落、线程退出 全部通过")
    print("=" * 64)


if __name__ == "__main__":
    main()
