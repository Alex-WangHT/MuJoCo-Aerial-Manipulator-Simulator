"""Robot（Multirotor + Manipulator 整机）组合与动力学耦合冒烟测试。

安装包后运行::

    python tests/test_robot.py

验证内容：
1. 组件经 Environment.attach_robot 两级 attach 组合编译成功，名称前缀化
   正确（uam/、uam/arm/），自省结果完整；
2. 闭环悬停 + 机械臂保持零位时，平台位置/姿态稳定（SO-ARM100 网格臂的
   质心偏离关节轴线，开环悬停物理上不可行，故悬停用 MultirotorController
   独立线程，经帧同步通道驱动）；
3. 摆动机械臂关节时，平台姿态出现可测扰动——证明臂-平台反作用耦合生效。
"""

from __future__ import annotations

import pathlib

import numpy as np

import UAMSim as ua


_BASE = pathlib.Path(__file__).resolve().parent.parent
_SCENE_XML = str(_BASE / "examples" / "models" / "environment.xml")
_MULTIROTOR_XML = str(_BASE / "examples" / "models" / "multirotor.xml")
_MANIPULATOR_XML = str(_BASE / "examples" / "models" / "Manipulator.xml")

DT = 0.001


class HoverController(ua.MultirotorController):
    """带姿态稳定的 PD 高度保持控制器（测试用）。"""

    def controller(self, feedback, targetPosition, mixer, mass, kp, kd, kp_att, kd_att):
        err = targetPosition - feedback.dronePosition
        vel = feedback.droneVelocity
        att = feedback.droneOrientation
        omega = feedback.droneAngularVelocity

        # 高度控制
        thrust = mass * 9.81 + kp * err[2] - kd * vel[2]

        # 姿态稳定（滚转/俯仰回零，yaw 阻尼）
        tau = np.array([
            -kp_att * att[0] - kd_att * omega[0],
            -kp_att * att[1] - kd_att * omega[1],
            -0.5 * att[2] - 0.1 * omega[2],
        ])

        return mixer @ np.array([thrust, tau[0], tau[1], tau[2]])


def main() -> None:
    print("=" * 64)
    print("Robot 整机组合冒烟测试（经 Environment 组装）")
    print("=" * 64)

    env = ua.Environment(_SCENE_XML)
    uam = env.attach_robot(ua.Multirotor(_MULTIROTOR_XML), ua.Manipulator(_MANIPULATOR_XML))
    model, data = env.model, env.data

    # ---------- 1. 结构与自省检查 ----------
    print("\n[1] 组合编译与名称自省")
    assert abs(model.opt.timestep - DT) < 1e-12, f"timestep 应为 {DT}，实际 {model.opt.timestep}"
    assert uam.multirotor.n_rotors == 6, f"旋翼数应为 6，实际 {uam.multirotor.n_rotors}"
    assert uam.manipulator.n_joints == 3, f"机械臂关节数应为 3，实际 {uam.manipulator.n_joints}"
    assert all(n.startswith("uam/arm/") for n in uam.manipulator.joint_names), (
        f"机械臂关节应带 uam/arm/ 前缀: {uam.manipulator.joint_names}"
    )

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

    # ---------- 2. 闭环悬停 + 机械臂零位保持（2 s，控制器独立线程） ----------
    print("\n[2] 闭环悬停 + 机械臂零位保持（2 s）")

    # 构造混控矩阵
    positions, yaw_coeffs, _ = uam.multirotor.get_rotor_geometry()
    alloc = np.vstack([
        np.ones(uam.multirotor.n_rotors),
        positions[:, 1],
        -positions[:, 0],
        yaw_coeffs,
    ])
    mixer = alloc.T @ np.linalg.inv(alloc @ alloc.T)

    drone_channel = ua.ControllerChannel()
    drone_ctrl = HoverController(
        uam.multirotor, drone_channel,
        targetPosition=[0.0, 0.0, 1.5],
        mixer=mixer,
        mass=total_mass,
        kp=4.0, kd=3.0,
        kp_att=6.0, kd_att=0.5,
    )
    drone_ctrl.start()
    uam.manipulator.set_actuator([0.0, 0.0, 0.0])

    def closed_loop(seconds: float) -> None:
        """手动物理循环（与 MujocoSimulation._frame 同构，主线程扮演仿真线程）。"""
        for _ in range(int(seconds / DT)):
            drone_channel.mailbox.publish(uam.multirotor.get_state())
            drone_channel.mailbox.wait_done()
            drone_channel.flush()
            env.step()

    try:
        closed_loop(2.0)

        state1 = uam.multirotor.get_state()
        euler1_deg = np.rad2deg(state1.droneOrientation)
        print(f"    t=2.0s 平台位置: {np.round(state1.dronePosition, 4)}")
        print(f"    t=2.0s 平台姿态[deg]: RPY={np.round(euler1_deg, 3)}")
        print(f"    t=2.0s 关节角[rad]: {np.round(uam.manipulator.get_joint_positions(), 4)}")
        assert abs(state1.dronePosition[2] - 1.5) < 0.1, "闭环悬停 2 s 后高度不应明显漂移"
        assert np.max(np.abs(euler1_deg)) < 5.0, "闭环悬停 2 s 后姿态应保持小角"
        assert np.max(np.abs(uam.manipulator.get_joint_positions())) < 0.05, "舵机应保持关节近零位"

        # ---------- 3. 摆动机械臂 -> 平台姿态应被反作用扰动 ----------
        print("\n[3] 机械臂摆动耦合测试（joint2 -> -0.6 rad，0.5 s，闭环悬停下）")
        uam.manipulator.set_actuator([0.0, -0.6, 0.0])
        max_deviation = 0.0
        for _ in range(int(0.5 / DT)):
            closed_loop(0.001)
            euler_deg = np.rad2deg(uam.multirotor.get_state().droneOrientation)
            max_deviation = max(max_deviation, float(np.max(np.abs(euler_deg - euler1_deg))))

        state2 = uam.multirotor.get_state()
        print(f"    关节角[rad]: {np.round(uam.manipulator.get_joint_positions(), 4)}")
        print(f"    平台姿态[deg]: RPY={np.round(np.rad2deg(state2.droneOrientation), 3)}")
        print(f"    姿态最大偏差: {max_deviation:.3f} deg")
        assert abs(uam.manipulator.get_joint_positions()[1] + 0.6) < 0.15, "舵机应跟踪到 -0.6 rad 附近"
        assert max_deviation > 1.0, (
            f"机械臂摆动未引起平台姿态扰动（最大偏差 {max_deviation:.3f} deg），耦合未生效"
        )
        assert max_deviation < 45.0, (
            f"扰动工况过于剧烈（最大偏差 {max_deviation:.3f} deg），演示应保持在未倾覆的有界区间"
        )

        ee_pos2, _ = uam.manipulator.get_ee_pose()
        ee_shift = float(np.linalg.norm(ee_pos2 - ee_pos))
        print(f"    末端位移: {ee_shift:.4f} m")
        assert ee_shift > 0.05, "机械臂末端应发生可见位移"
    finally:
        drone_channel.close()
        drone_ctrl.join(timeout=2.0)
    assert not drone_ctrl.is_alive(), "控制器线程应随通道关闭而退出"

    print("\n" + "=" * 64)
    print("PASS: 组合编译、名称自省、悬停平衡、臂-平台耦合 全部通过")
    print("=" * 64)


if __name__ == "__main__":
    main()
