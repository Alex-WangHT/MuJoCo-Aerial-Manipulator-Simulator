"""MultirotorController / ManipulatorController 闭环测试。

直接运行：
    .venv/Scripts/python tests/test_controllers.py

验证内容：
1. 混控器构建：分配矩阵伪逆形状、推力范围、悬停推力基线；
2. MultirotorController 位置闭环：从 1.5 m 起飞到 2.0 m 并定点，姿态保持小角；
3. ManipulatorController 关节模式：目标角跟踪；
4. ManipulatorController 末端模式：DLS 逆解收敛到世界系目标点（平台闭环悬停下）；
5. 线程集成：两个控制器经 MujocoSimulation 注入后自动 bind，无头运行正常。
"""

from __future__ import annotations

import pathlib
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src import (
    AerialManipulator,
    Environment,
    ManipulatorController,
    MujocoSimulation,
    MultirotorController,
    TelemetryBuffer,
)

DT = 0.001


def build_scene():
    """构建场景 + 机器人（与 MujocoSimulation._build 相同的组装路径）。"""
    env = Environment()
    uam = AerialManipulator(compileModel=False)
    env.attach_robot(uam)
    return env, uam


def run_closed_loop(env, uam, drone_ctrl, arm_ctrl, seconds):
    """手动控制循环（与 MujocoSimulation._frame 同构，无线程）。"""
    steps = int(seconds / DT)
    for _ in range(steps):
        sensor = uam.multirotor.get_state()
        drone_ctrl.setSensorData(sensor)
        u = drone_ctrl.getControlInput().u
        uam.multirotor.set_thrusts(u)
        if arm_ctrl is not None:
            arm_ctrl.update()
        env.step()


def main() -> None:
    print("=" * 64)
    print("MultirotorController / ManipulatorController 闭环测试")
    print("=" * 64)

    # ------------------------------------------------ [1] 混控器构建
    print("\n[1] 混控器构建（bind 自省）")
    env, uam = build_scene()
    drone_ctrl = MultirotorController(targetPosition=[0.0, 0.0, 1.5])
    drone_ctrl.bind(uam.multirotor)
    assert drone_ctrl._alloc_pinv.shape == (uam.multirotor.n_rotors, 4), "分配伪逆形状错误"
    assert drone_ctrl._mass > 1.0, "整机质量应已读取"
    hover_u = drone_ctrl._u
    expected_hover = drone_ctrl._mass * 9.81 / uam.multirotor.n_rotors
    assert abs(hover_u[0] - expected_hover) < 1e-6, "初始 u 应为均分悬停推力"
    print(f"    整机质量: {drone_ctrl._mass:.3f} kg, 悬停推力基线: {hover_u[0]:.3f} N/rotor")
    print(f"    推力范围: [{drone_ctrl._ctrl_min[0]:.1f}, {drone_ctrl._ctrl_max[0]:.1f}] N")

    # ---------------------------------- [2] 位置闭环：1.5 m -> 2.0 m 定点
    print("\n[2] MultirotorController 位置闭环（目标 z=2.0 m，4 s）")
    drone_ctrl.set_target_position([0.0, 0.0, 2.0])
    run_closed_loop(env, uam, drone_ctrl, None, 4.0)
    state = uam.multirotor.get_state()
    print(f"    末端位置: {np.round(state.dronePosition, 3)}")
    print(f"    姿态[deg]: RPY={np.round(np.degrees(state.droneOrientation), 2)}")
    assert abs(state.dronePosition[2] - 2.0) < 0.1, f"高度未收敛: {state.dronePosition[2]}"
    assert np.linalg.norm(state.dronePosition[:2]) < 0.1, "水平位置漂移过大"
    assert np.all(np.abs(state.droneOrientation[:2]) < np.radians(10)), "姿态角过大"

    # ---------------------------------- [3] 关节模式：目标角跟踪
    print("\n[3] ManipulatorController 关节模式（[0.4, -0.3] rad，1.5 s）")
    arm_ctrl = ManipulatorController()
    arm_ctrl.bind(uam.manipulator)
    arm_ctrl.set_target_joints([0.4, -0.3])
    run_closed_loop(env, uam, drone_ctrl, arm_ctrl, 1.5)
    q = uam.manipulator.get_joint_positions()
    print(f"    关节角[rad]: {np.round(q, 3)}（目标 [0.4, -0.3]）")
    assert np.allclose(q, [0.4, -0.3], atol=0.1), f"关节角未收敛: {q}"
    state = uam.multirotor.get_state()
    assert abs(state.dronePosition[2] - 2.0) < 0.15, "摆臂时平台高度保持失败"

    # -------------------------- [4] 末端模式：DLS 逆解收敛到世界系目标点
    print("\n[4] ManipulatorController 末端模式（EE +[0.08, 0, 0.03] m，3 s）")
    arm_ctrl.set_target_joints(uam.manipulator.get_joint_positions())  # 从当前姿态出发
    ee_start = uam.manipulator.get_ee_pose()[0]
    ee_goal = ee_start + np.array([0.08, 0.0, 0.03])
    arm_ctrl.set_target_ee(ee_goal)
    assert arm_ctrl.mode == "ee"
    run_closed_loop(env, uam, drone_ctrl, arm_ctrl, 3.0)
    ee_final = uam.manipulator.get_ee_pose()[0]
    ee_err = float(np.linalg.norm(ee_final - ee_goal))
    print(f"    EE 目标: {np.round(ee_goal, 3)}, 实际: {np.round(ee_final, 3)}, 误差: {ee_err * 1000:.1f} mm")
    assert ee_err < 0.01, f"末端未收敛，误差 {ee_err * 1000:.1f} mm"

    # -------------------------- [5] 线程集成：注入仿真自动 bind + 无头运行
    print("\n[5] MujocoSimulation 线程集成（双控制器注入，无头 1.5 s）")
    shutdown = threading.Event()
    telemetry = TelemetryBuffer()
    drone_ctrl2 = MultirotorController(targetPosition=[0.5, 0.0, 1.8])
    arm_ctrl2 = ManipulatorController(jointTargets=[0.2, 0.2])
    sim = MujocoSimulation(
        shutdown,
        telemetryBuffer=telemetry,
        droneController=drone_ctrl2,
        manipulatorController=arm_ctrl2,
        useViewer=False,
    )
    sim.start()
    time.sleep(1.5)
    shutdown.set()
    sim.join(timeout=3.0)
    assert not sim.is_alive(), "线程未正常退出"
    assert drone_ctrl2._bound and arm_ctrl2._bound, "控制器应由仿真自动 bind"
    t, pos, _ = telemetry.snapshot()
    assert len(t) > 500, "遥测样本过少"
    print(f"    遥测样本: {len(t)}, 最终位置: {np.round(pos[-1], 3)}")
    assert abs(pos[-1][2] - 1.8) < 0.2, "线程闭环高度异常"

    # -------------------------- [6] 用户自定义控制器（只重写一个控制函数）
    print("\n[6] 用户自定义控制器子类（继承基类，只重写控制函数）")

    class UserHoverController(MultirotorController):
        """最小自定义示例：整个子类只需重写 compute_control 一个方法，
        装配（bind）、协议、混控矩阵、推力截断全部由基类完成。"""

        def compute_control(self, sensor):
            e = self._target_position - sensor.dronePosition
            thrust = self._mass * (
                self._gravity + 3.0 * e[2] - 2.5 * sensor.droneVelocity[2]
            )
            torque = (
                -np.array([4.0, 4.0, 0.8]) * sensor.droneOrientation
                - 0.3 * sensor.droneAngularVelocity
            )
            return self._alloc_pinv @ np.array([thrust, *torque])

    class UserArmPoseController(ManipulatorController):
        """最小自定义示例：只重写 compute_joint_targets 一个方法。"""

        def compute_joint_targets(self):
            return np.array([0.3, -0.2])

    env2, uam2 = build_scene()
    user_drone = UserHoverController(targetPosition=[0.0, 0.0, 1.8])
    user_drone.bind(uam2.multirotor)
    user_arm = UserArmPoseController()
    user_arm.bind(uam2.manipulator)
    run_closed_loop(env2, uam2, user_drone, user_arm, 3.0)
    state = uam2.multirotor.get_state()
    q = uam2.manipulator.get_joint_positions()
    print(f"    平台位置: {np.round(state.dronePosition, 3)}（目标 z=1.8）")
    print(f"    姿态[deg]: RPY={np.round(np.degrees(state.droneOrientation), 2)}")
    print(f"    关节角[rad]: {np.round(q, 3)}（目标 [0.3, -0.2]）")
    assert abs(state.dronePosition[2] - 1.8) < 0.1, "自定义控制器高度未收敛"
    assert np.all(np.abs(state.droneOrientation[:2]) < np.radians(10)), "自定义控制器姿态过大"
    assert np.allclose(q, [0.3, -0.2], atol=0.1), "自定义臂控制器关节未收敛"

    # 语法级冻结保护：重写除控制函数外的任何基类方法 -> 类创建即抛 TypeError
    frozen_hits = 0
    try:
        class BadBindController(MultirotorController):
            def bind(self, multirotor):
                pass
    except TypeError:
        frozen_hits += 1
    try:
        class BadInitController(MultirotorController):
            def __init__(self):
                pass
    except TypeError:
        frozen_hits += 1
    try:
        class BadUpdateController(ManipulatorController):
            def update(self):
                pass
    except TypeError:
        frozen_hits += 1
    try:
        class BadDlsController(ManipulatorController):
            def _ee_dls_step(self):
                pass
    except TypeError:
        frozen_hits += 1
    assert frozen_hits == 4, f"冻结保护未生效（仅拦截 {frozen_hits}/4）"
    print("    冻结保护生效：重写 bind / __init__ / update / _ee_dls_step 均抛 TypeError")

    print("\n" + "=" * 64)
    print("PASS: 混控构建、位置闭环、关节/末端控制、线程集成、自定义子类 全部通过")
    print("=" * 64)


if __name__ == "__main__":
    main()
