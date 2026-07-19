"""AerialManipulator 组合与动力学耦合冒烟测试。

直接运行：
    .venv/Scripts/python tests/test_aerial_manipulator.py

验证内容：
1. 两份 MJCF 经 mjSpec.attach 组合编译成功，名称前缀化正确，自省结果完整；
2. 悬停推力 + 机械臂保持零位时，平台近似静止（反扭矩平衡、挂载几何对中）；
3. 摆动机械臂关节时，平台姿态出现明显扰动——证明臂-平台反作用耦合生效。
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src import AerialManipulator

DT = 0.001


def main() -> None:
    print("=" * 64)
    print("AerialManipulator 组合冒烟测试")
    print("=" * 64)

    uam = AerialManipulator()
    model, data = uam.model, uam.data

    # ---------- 1. 结构与自省检查 ----------
    print("\n[1] 组合编译与名称自省")
    assert abs(model.opt.timestep - DT) < 1e-12, f"timestep 应为 {DT}，实际 {model.opt.timestep}"
    assert uam.multirotor.n_rotors == 6, f"旋翼数应为 6，实际 {uam.multirotor.n_rotors}"
    assert uam.manipulator.n_joints == 2, f"机械臂关节数应为 2，实际 {uam.manipulator.n_joints}"
    assert all(n.startswith("arm/") for n in uam.manipulator.joint_names), "机械臂关节应带 arm/ 前缀"

    total_mass = float(model.body_mass.sum())
    print(f"    整机总质量: {total_mass:.3f} kg")
    print(f"    旋翼(actuator id): {[(n, i) for n, i in uam.multirotor.rotors.items()]}")
    print(f"    机械臂关节: {uam.manipulator.joint_names}")
    print(f"    机械臂传感器: {uam.manipulator.sensor_names}")

    mount_pos, _ = uam.multirotor.get_mount_pose()
    assert abs(mount_pos[0]) < 1e-9 and abs(mount_pos[1]) < 1e-9, "挂载点应位于机体中轴线上"
    assert abs(mount_pos[2] - 1.45) < 1e-9, f"挂载点世界高度应为 1.45 m，实际 {mount_pos[2]:.4f}"
    print(f"    挂载点世界位置: {np.round(mount_pos, 4)}（机体系 pos=0 0 -0.05，固连正确）")

    ee_pos, _ = uam.manipulator.get_ee_pose()
    print(f"    末端初始世界位置: {np.round(ee_pos, 4)}")

    imu = uam.multirotor.get_imu()
    assert imu["gyro"].shape == (3,) and imu["accel"].shape == (3,), "IMU 维度应为 3"
    print(f"    IMU 初始读数: gyro={np.round(imu['gyro'], 4)}, accel={np.round(imu['accel'], 4)}")

    # ---------- 2. 悬停 + 机械臂零位保持（1 s） ----------
    print("\n[2] 悬停 + 机械臂零位保持（1 s）")
    u_hover = uam.hover_thrust()
    print(f"    悬停推力估计: {u_hover:.3f} N/rotor")
    uam.multirotor.set_thrusts(np.full(uam.multirotor.n_rotors, u_hover))
    uam.manipulator.set_joint_targets([0.0, 0.0])
    uam.step(int(1.0 / DT))

    state1 = uam.multirotor.get_state()
    euler1_deg = np.rad2deg(state1.droneOrientation)
    print(f"    t=1.0s 平台位置: {np.round(state1.dronePosition, 4)}")
    print(f"    t=1.0s 平台姿态[deg]: RPY={np.round(euler1_deg, 3)}")
    print(f"    t=1.0s 关节角[rad]: {np.round(uam.manipulator.get_joint_positions(), 4)}")
    assert abs(state1.dronePosition[2] - 1.5) < 0.3, "悬停 1 s 后高度不应明显漂移"
    assert np.max(np.abs(euler1_deg)) < 5.0, "悬停 1 s 后姿态不应明显漂移（反扭矩应自平衡）"
    assert np.max(np.abs(uam.manipulator.get_joint_positions())) < 0.05, "舵机应保持关节近零位"

    # ---------- 3. 摆动机械臂 -> 平台姿态应被反作用扰动 ----------
    print("\n[3] 机械臂摆动耦合测试（joint1 -> 0.5 rad，0.3 s）")
    uam.manipulator.set_joint_targets([0.5, 0.0])
    max_deviation = 0.0
    for _ in range(int(0.3 / DT)):
        uam.step()
        euler_deg = np.rad2deg(uam.multirotor.get_state().droneOrientation)
        max_deviation = max(max_deviation, float(np.max(np.abs(euler_deg - euler1_deg))))

    state2 = uam.multirotor.get_state()
    print(f"    关节角[rad]: {np.round(uam.manipulator.get_joint_positions(), 4)}")
    print(f"    平台姿态[deg]: RPY={np.round(np.rad2deg(state2.droneOrientation), 3)}")
    print(f"    姿态最大偏差: {max_deviation:.3f} deg")
    assert abs(uam.manipulator.get_joint_positions()[0] - 0.5) < 0.15, "舵机应跟踪到 0.5 rad 附近"
    assert max_deviation > 3.0, (
        f"机械臂摆动未引起平台姿态扰动（最大偏差 {max_deviation:.3f} deg），耦合未生效"
    )
    assert max_deviation < 45.0, (
        f"扰动工况过于剧烈（最大偏差 {max_deviation:.3f} deg），演示应保持在未倾覆的有界区间"
    )

    ee_pos2, _ = uam.manipulator.get_ee_pose()
    ee_shift = float(np.linalg.norm(ee_pos2 - ee_pos))
    print(f"    末端位移: {ee_shift:.4f} m")
    assert ee_shift > 0.05, "机械臂末端应发生可见位移"

    print("\n" + "=" * 64)
    print("PASS: 组合编译、名称自省、悬停平衡、臂-平台耦合 全部通过")
    print("=" * 64)


if __name__ == "__main__":
    main()
