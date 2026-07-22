"""临时验证脚本：合并重构（Environment.attach_robot 吸收 AerialManipulator）。

注意：HEAD 提交已删除 src/MultirotorController.py / src/ManipulatorController.py
（用户进行中的控制器重写），这里用 stub 绕过 src/__init__ 对它们的导入，
只验证不依赖控制器的部分。验证完成后删除本文件。
"""

from __future__ import annotations

import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# ---- stub 缺失的控制器模块（HEAD 已删除，待用户重写） ----
for name in ("MultirotorController", "ManipulatorController"):
    module = types.ModuleType(f"src.{name}")
    setattr(module, name, type(name, (), {}))
    sys.modules[f"src.{name}"] = module

import threading
import time

import numpy as np

from src import Environment, Manipulator, MujocoSimulation, Multirotor, Robot, TelemetryBuffer


def main() -> None:
    # [1] attach + compile + 前缀 + 句柄
    env = Environment()
    uam = env.attach_robot(Multirotor(), Manipulator())
    assert isinstance(uam, Robot) and uam is env.robots[-1]
    assert uam.has_manipulator
    assert uam.model is env.model and uam.data is env.data
    assert "uam/rotor1" in uam.multirotor.rotors
    assert all(n.startswith("uam/arm/") for n in uam.manipulator.joint_names)
    print("[1] attach/compile/前缀/句柄 OK; robots:", len(env.robots))

    # [2] 出生位姿 + 挂载点
    state = uam.multirotor.get_state()
    assert np.allclose(state.dronePosition, [0, 0, 1.5], atol=1e-9)
    mount_pos, _ = uam.multirotor.get_mount_pose()
    assert abs(mount_pos[2] - 1.45) < 1e-9
    print("[2] 出生位姿 OK:", np.round(state.dronePosition, 3))

    # [3] 障碍物 + 悬停推力
    obstacles = env.get_obstacle_positions()
    assert set(obstacles) == {"obstacle_pillar_a", "obstacle_pillar_b", "obstacle_box"}
    hover = uam.hover_thrust()
    expected = float(env.model.body_mass.sum()) * 9.81 / uam.multirotor.n_rotors
    assert abs(hover - expected) < 1e-9 and hover > 0
    print(f"[3] 障碍物 OK; hover_thrust = {hover:.3f} N/rotor")

    # [4] 重复挂载已绑定组件 -> ValueError
    try:
        env.attach_robot(uam.multirotor)
        raise AssertionError("应拒绝重复挂载")
    except ValueError:
        print("[4] 重复挂载拒绝 OK")

    # [5] 纯多旋翼构型
    env2 = Environment()
    uam2 = env2.attach_robot(Multirotor())
    assert not uam2.has_manipulator and uam2.manipulator is None
    assert "uam/rotor6" in uam2.multirotor.rotors
    print("[5] 纯多旋翼构型 OK")

    # [6] 舵机写入 + 末端位移（机械臂合并进场景后可控）
    before = uam.manipulator.get_ee_pose()[0].copy()
    uam.manipulator.set_actuator([0.0, -0.6, 0.0])
    uam.multirotor.set_actuator(np.full(uam.multirotor.n_rotors, hover))
    for _ in range(500):
        env.step()
    after = uam.manipulator.get_ee_pose()[0]
    assert np.linalg.norm(after - before) > 0.05
    assert np.isfinite(env.data.qpos).all()
    print(f"[6] 舵机驱动末端位移 OK: {np.linalg.norm(after - before):.3f} m")

    # [7] 零推力下落 -> 与地板接触
    env.reset()
    uam.multirotor.set_actuator(np.zeros(uam.multirotor.n_rotors))
    contacts = []
    for _ in range(1500):
        env.step()
        contacts = env.get_robot_contacts()
        if contacts:
            break
    assert contacts and any("floor" in (n1, n2) for n1, n2, _ in contacts)
    print("[7] 下落接触 OK:", contacts[0][:2])

    # [8] MujocoSimulation 新签名端到端（无控制器，固定推力 0，无头 0.3 s）
    shutdown = threading.Event()
    telemetry = TelemetryBuffer()
    env3 = Environment()
    sim = MujocoSimulation(shutdown, env3, Multirotor(), Manipulator(),
                           telemetryBuffer=telemetry, useViewer=False, fixedRotorThrust=0.0)
    sim.start()
    assert sim.wait_ready(timeout=10.0)
    assert isinstance(sim.uam, Robot) and sim.uam.has_manipulator
    assert sim.drone_channel is not None and sim.arm_channel is not None
    sim.start_physics()
    time.sleep(0.3)
    shutdown.set()
    sim.join(timeout=3.0)
    assert not sim.is_alive()
    t, pos, _ = telemetry.snapshot()
    assert len(t) > 50 and pos[-1][2] < 1.5
    print(f"[8] MujocoSimulation 端到端 OK: {len(t)} 帧, 末端高度 {pos[-1][2]:.3f} m（下落）")

    print("\nPASS: 合并重构验证全部通过（控制器相关测试待控制器重写后运行）")


if __name__ == "__main__":
    main()
