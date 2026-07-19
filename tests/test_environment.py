"""Environment（MuJoCo 场景环境）组合与交互测试。

直接运行：
    .venv/Scripts/python tests/test_environment.py

验证内容：
1. Environment 加载场景、AerialManipulator（未编译模式）attach 进场景统一编译；
2. 机器人名称带场景前缀（uam/、uam/arm/），出生位姿正确；
3. 障碍物按 obstacle_ 前缀自被发现且位置与 MJCF 一致；
4. 场景中悬停 0.5 s 高度保持；
5. 零推力下落后机器人与地板产生接触（get_robot_contacts 非空）。
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src import AerialManipulator, Environment


def main() -> None:
    print("=" * 64)
    print("Environment 场景组合与交互测试")
    print("=" * 64)

    # ---------- 1. 组合编译与前缀检查 ----------
    print("\n[1] 场景组合编译")
    env = Environment()
    uam = AerialManipulator(compileModel=False)
    env.attach_robot(uam)

    assert uam.model is env.model, "机器人应共享场景模型"
    rotor_names = list(uam.multirotor.rotors)
    assert "uam/rotor1" in rotor_names, f"旋翼应带 uam/ 前缀: {rotor_names}"
    assert all(n.startswith("uam/arm/") for n in uam.manipulator.joint_names), (
        f"机械臂关节应带 uam/arm/ 前缀: {uam.manipulator.joint_names}"
    )
    print(f"    旋翼: {rotor_names}")
    print(f"    机械臂关节: {uam.manipulator.joint_names}")
    print(f"    场景 geom 总数: {env.model.ngeom}, 接触初始数: {env.data.ncon}")

    # ---------- 2. 出生位姿 ----------
    print("\n[2] 出生位姿")
    state = uam.multirotor.get_state()
    assert np.allclose(state.dronePosition, [0, 0, 1.5], atol=1e-9), (
        f"出生位置应为 (0,0,1.5)，实际 {state.dronePosition}"
    )
    mount_pos, _ = uam.multirotor.get_mount_pose()
    assert abs(mount_pos[2] - 1.45) < 1e-9, f"挂载点高度应为 1.45 m，实际 {mount_pos[2]:.4f}"
    print(f"    平台位置: {np.round(state.dronePosition, 4)}, 挂载点高度: {mount_pos[2]:.3f} m")

    # ---------- 3. 障碍物自省 ----------
    print("\n[3] 障碍物自省")
    obstacle_positions = env.get_obstacle_positions()
    assert set(obstacle_positions) == {"obstacle_pillar_a", "obstacle_pillar_b", "obstacle_box"}, (
        f"应发现 3 个障碍物: {sorted(obstacle_positions)}"
    )
    assert np.allclose(obstacle_positions["obstacle_pillar_a"], [1.5, 0.5, 0.5], atol=1e-9)
    assert np.allclose(obstacle_positions["obstacle_box"], [0.3, -1.5, 0.2], atol=1e-9)
    for name, pos in sorted(obstacle_positions.items()):
        print(f"    {name}: {np.round(pos, 3)}")

    # ---------- 4. 场景中悬停 0.5 s ----------
    print("\n[4] 场景中悬停（0.5 s）")
    uam.multirotor.set_thrusts(np.full(uam.multirotor.n_rotors, uam.hover_thrust()))
    uam.manipulator.set_joint_targets([0.0, 0.0])
    env.step(500)
    state = uam.multirotor.get_state()
    print(f"    t=0.5s 高度: {state.dronePosition[2]:.4f} m, 姿态[deg]: {np.round(np.rad2deg(state.droneOrientation), 2)}")
    assert abs(state.dronePosition[2] - 1.5) < 0.1, "悬停 0.5 s 高度应保持"

    # ---------- 5. 下落接触交互 ----------
    print("\n[5] 零推力下落 -> 机器人与地板接触")
    uam.multirotor.set_thrusts(np.zeros(uam.multirotor.n_rotors))
    robot_contacts = []
    for _ in range(1500):
        env.step()
        robot_contacts = env.get_robot_contacts()
        if robot_contacts:
            break
    assert robot_contacts, "下落 1.5 s 内机器人应与地板发生接触"
    contact_pairs = {(n1, n2) for n1, n2, _ in robot_contacts}
    assert any("floor" in pair for pair in contact_pairs), "接触对中应包含 floor"
    state = uam.multirotor.get_state()
    print(f"    首次接触于 t={state.timestamp:.2f} s, 高度 {state.dronePosition[2]:.3f} m")
    for name1, name2, pos in robot_contacts[:6]:
        print(f"    接触: {name1} <-> {name2} @ {np.round(pos, 3)}")

    print("\n" + "=" * 64)
    print("PASS: 场景组合、出生位姿、障碍物自省、悬停、接触交互 全部通过")
    print("=" * 64)


if __name__ == "__main__":
    main()
