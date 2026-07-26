"""UAMSim 综合演示脚本。

合并了简单悬停与完整集成演示两种场景：

1. 简单悬停（默认）：纯多旋翼 PD 高度保持，3 秒
   python examples/demo.py

2. 完整集成演示（--full）：多旋翼级联 PID + 机械臂各关节 ±1 rad 摆动，9 秒
   python examples/demo.py --full
   python examples/demo.py --full --headless   # 无头 5x 加速
"""

from __future__ import annotations

import argparse
import pathlib
import threading
import time

import numpy as np

import UAMSim as ua


# ---------------------------------------------------------------------------
# 模型路径（本示例附带）
# ---------------------------------------------------------------------------

_MODELS_DIR = pathlib.Path(__file__).resolve().parent / "models"
_SCENE_XML = str(_MODELS_DIR / "environment.xml")
_MULTIROTOR_XML = str(_MODELS_DIR / "multirotor.xml")
_MANIPULATOR_XML = str(_MODELS_DIR / "Manipulator.xml")


# ---------------------------------------------------------------------------
# 简单悬停控制器
# ---------------------------------------------------------------------------

class HoverController(ua.MultirotorController):
    """简单 PD 高度保持控制器。"""

    def controller(self, feedback, target_pos, kp, kd, hover):
        err = target_pos - feedback.dronePosition
        vel = feedback.droneVelocity
        thrust = hover + kp * err[2] - kd * vel[2]
        return np.full(self._multirotor.n_rotors, thrust)


# ---------------------------------------------------------------------------
# 级联 PID 多旋翼控制器（完整集成演示用）
# ---------------------------------------------------------------------------

class CascadedPidMultirotor(ua.MultirotorController):
    """级联 PID 多旋翼控制器。

    外环（位置）：位置 PID -> 期望加速度 -> 期望总推力 + 期望滚转/俯仰；
    内环（姿态）：姿态 PID（含积分，消除机械臂重力矩引起的稳态倾角）
    -> 期望三轴力矩；混控矩阵 mixer（N x 4，伪逆预分配）把
    [总推力, τx, τy, τz] 映射为各旋翼推力。
    """

    def controller(self, feedback, mass, gravity, pos_ref, gains, mixer, state):
        pos = feedback.dronePosition
        vel = feedback.droneVelocity
        att = feedback.droneOrientation
        omega = feedback.droneAngularVelocity
        dt = max(feedback.timestep, 1e-6)

        # 外环：位置 PID -> 期望加速度
        err = pos_ref - pos
        state["int_z"] = np.clip(
            state["int_z"] + err[2] * dt, -gains["int_z_limit"], gains["int_z_limit"]
        )
        ax = gains["kp_xy"] * err[0] - gains["kd_xy"] * vel[0]
        ay = gains["kp_xy"] * err[1] - gains["kd_xy"] * vel[1]
        az = (gains["kp_z"] * err[2] - gains["kd_z"] * vel[2]
              + gains["ki_z"] * state["int_z"])
        thrust = mass * (gravity + az)

        # 期望姿态
        psi = att[2]
        roll_des = np.clip((np.sin(psi) * ax - np.cos(psi) * ay) / gravity, -0.3, 0.3)
        pitch_des = np.clip((np.cos(psi) * ax + np.sin(psi) * ay) / gravity, -0.3, 0.3)
        att_des = np.array([roll_des, pitch_des, 0.0])

        # 内环：姿态 PID -> 期望力矩
        att_err = att_des - att
        state["int_att"] = np.clip(
            state["int_att"] + att_err * dt, -gains["int_att_limit"], gains["int_att_limit"]
        )
        tau = (gains["kp_att"] * att_err - gains["kd_att"] * omega
               + gains["ki_att"] * state["int_att"])

        return mixer @ np.array([thrust, tau[0], tau[1], tau[2]])


# ---------------------------------------------------------------------------
# 机械臂摆动控制器（完整集成演示用）
# ---------------------------------------------------------------------------

class SwingJointController(ua.ManipulatorController):
    """机械臂各关节 ±1 rad 正弦摆动。"""

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = min(max(x, 0.0), 1.0)
        return x * x * (3.0 - 2.0 * x)

    def controller(self, feedback, center, period, swing_stop, state):
        np.minimum(state["q_min"], feedback.jointPositions, out=state["q_min"])
        np.maximum(state["q_max"], feedback.jointPositions, out=state["q_max"])
        t = feedback.timestamp
        envelope = self._smoothstep(t)
        if swing_stop is not None:
            envelope = min(envelope, self._smoothstep(swing_stop - t))
        return envelope * (center + np.sin(2.0 * np.pi * t / period))


def build_mixer(multirotor) -> np.ndarray:
    """由旋翼几何构造混控伪逆矩阵 (N, 4)：[F, τx, τy, τz] -> u。"""
    positions, yaw_coeffs, _ = multirotor.get_rotor_geometry()
    alloc = np.vstack([
        np.ones(multirotor.n_rotors),
        positions[:, 1],
        -positions[:, 0],
        yaw_coeffs,
    ])
    return alloc.T @ np.linalg.inv(alloc @ alloc.T)


# ---------------------------------------------------------------------------
# 场景 1：简单纯多旋翼悬停
# ---------------------------------------------------------------------------

def run_hover_simple(headless: bool = False, duration: float = 3.0) -> None:
    print("=" * 64)
    print("UAMSim 演示 —— 纯多旋翼悬停（简单版）")
    print(f"模式: {'无头' if headless else 'GUI'}, 时长: {duration:.1f} s")
    print("=" * 64)

    shutdown = threading.Event()
    env = ua.Environment(_SCENE_XML)
    telemetry = ua.TelemetryBuffer()
    sim = ua.MujocoSimulation(
        shutdown, env, ua.Multirotor(_MULTIROTOR_XML), None,
        telemetry_buffer=telemetry,
        use_viewer=not headless,
        real_time_factor=5.0 if headless else 1.0,
    )
    sim.start()
    assert sim.wait_ready(timeout=10.0), "场景编译超时"

    ctrl = HoverController(
        sim.uam.multirotor, sim.drone_channel, shutdown,
        target_pos=np.array([0.0, 0.0, 1.5]),
        kp=4.0, kd=3.0, hover=sim.uam.hover_thrust(),
    )
    ctrl.start()
    sim.start_physics()

    time.sleep(duration)

    shutdown.set()
    sim.join(timeout=5.0)
    ctrl.join(timeout=2.0)

    t_arr, pos_arr, eul_arr = telemetry.snapshot()
    print(f"\n仿真时长: {t_arr[-1]:.2f} s")
    print(f"最终位置: {np.round(pos_arr[-1], 3)}")
    print(f"最终姿态[deg]: {np.round(np.rad2deg(eul_arr[-1]), 2)}")
    print(f"高度误差: {abs(pos_arr[-1][2] - 1.5):.3f} m")
    print("DONE: 简单悬停演示结束")


# ---------------------------------------------------------------------------
# 场景 2：完整集成演示（级联 PID + 机械臂摆动）
# ---------------------------------------------------------------------------

def run_full_demo(headless: bool = False) -> None:
    SPAWN_Z = 1.5
    SWING_PERIOD = 4.0
    SWING_STOP = 7.0
    SIM_TIME = 9.0
    TILT_LIMIT = 0.20
    Z_ERR_LIMIT = 0.3
    XY_LIMIT = 0.8

    print("=" * 64)
    print("UAMSim 演示 —— 完整集成（级联 PID + 机械臂 ±1 rad 摆动）")
    print(f"模式: {'无头 5x' if headless else 'GUI 实时 1x'}")
    print("=" * 64)

    shutdown = threading.Event()
    telemetry = ua.TelemetryBuffer(max_samples=20000)
    env = ua.Environment(_SCENE_XML)
    sim = ua.MujocoSimulation(
        shutdown, env, ua.Multirotor(_MULTIROTOR_XML), ua.Manipulator(_MANIPULATOR_XML),
        telemetry_buffer=telemetry, use_viewer=not headless,
        real_time_factor=5.0 if headless else 1.0,
    )
    sim.start()
    assert sim.wait_ready(timeout=10.0), "场景编译超时"
    uam = sim.uam

    # 多旋翼：级联 PID
    mass = float(uam.model.body_mass.sum())
    gains = {
        "kp_xy": 2.0, "kd_xy": 2.5,
        "kp_z": 4.0, "kd_z": 3.0, "ki_z": 1.5, "int_z_limit": 1.0,
        "kp_att": np.array([3.0, 3.0, 1.0]),
        "kd_att": np.array([0.30, 0.30, 0.10]),
        "ki_att": np.array([0.8, 0.8, 0.0]),
        "int_att_limit": np.array([0.6, 0.6, 0.0]),
    }
    drone_ctrl = CascadedPidMultirotor(
        uam.multirotor, sim.drone_channel, shutdown,
        mass=mass, gravity=9.81, pos_ref=np.array([0.0, 0.0, SPAWN_Z]),
        gains=gains, mixer=build_mixer(uam.multirotor),
        state={"int_z": 0.0, "int_att": np.zeros(3)},
    )

    # 机械臂：各关节 ±1 rad 正弦摆动
    q0 = uam.manipulator.get_joint_positions()
    center = np.array([0.0, -1.0, 1.0])
    arm_ctrl = SwingJointController(
        uam.manipulator, sim.arm_channel, shutdown,
        center=center, period=SWING_PERIOD, swing_stop=SWING_STOP,
        state={"q_min": q0.copy(), "q_max": q0.copy()},
    )

    drone_ctrl.start()
    arm_ctrl.start()
    sim.start_physics()

    # 全程连续监控
    monitor = {"max_tilt": 0.0, "max_z_err": 0.0, "max_xy": 0.0}
    sim_time = 0.0
    deadline = time.time() + 120.0
    while sim_time < SIM_TIME and time.time() < deadline and sim.is_alive():
        t_arr, pos_arr, eul_arr = telemetry.snapshot()
        if len(t_arr):
            sim_time = float(t_arr[-1])
            monitor["max_tilt"] = float(np.max(np.abs(eul_arr[:, :2])))
            monitor["max_z_err"] = float(np.max(np.abs(pos_arr[:, 2] - SPAWN_Z)))
            monitor["max_xy"] = float(np.max(np.linalg.norm(pos_arr[:, :2], axis=1)))
            if monitor["max_tilt"] >= TILT_LIMIT:
                raise RuntimeError(
                    f"t={sim_time:.2f}s 倾角 {monitor['max_tilt']:.4f} rad 超限 {TILT_LIMIT} rad"
                )
            if monitor["max_z_err"] >= Z_ERR_LIMIT:
                raise RuntimeError(
                    f"t={sim_time:.2f}s 高度误差 {monitor['max_z_err']:.3f} m 超限 {Z_ERR_LIMIT} m"
                )
            if monitor["max_xy"] >= XY_LIMIT:
                raise RuntimeError(
                    f"t={sim_time:.2f}s 水平漂移 {monitor['max_xy']:.3f} m 超限 {XY_LIMIT} m"
                )
        time.sleep(0.1)

    if sim_time < SIM_TIME and not sim.is_alive():
        shutdown.set()
        sim.join(timeout=5.0)
        raise RuntimeError(
            f"仿真提前结束（viewer 被关闭？）：仿真时间 {sim_time:.2f} s < {SIM_TIME} s"
        )

    shutdown.set()
    sim.join(timeout=5.0)
    drone_ctrl.join(timeout=2.0)
    arm_ctrl.join(timeout=2.0)

    final_state = uam.multirotor.get_state()
    q_end = uam.manipulator.get_joint_positions()
    sweep = arm_ctrl.params["state"]["q_max"] - arm_ctrl.params["state"]["q_min"]
    tilt_end = max(abs(final_state.droneOrientation[0]), abs(final_state.droneOrientation[1]))
    z_err = abs(final_state.dronePosition[2] - SPAWN_Z)
    xy_drift = float(np.linalg.norm(final_state.dronePosition[:2]))

    print(f"\n仿真时间: {sim_time:.2f} s")
    print(f"全程最大倾角: {monitor['max_tilt']:.4f} rad ({np.degrees(monitor['max_tilt']):.2f} deg), "
          f"最大高度误差: {monitor['max_z_err']:.3f} m, 最大水平漂移: {monitor['max_xy']:.3f} m")
    print(f"结束姿态: {np.round(final_state.droneOrientation, 4)}, "
          f"结束位置: {np.round(final_state.dronePosition, 3)}")
    print(f"关节实际摆幅（峰峰值）: {np.round(sweep, 3)} rad, 结束关节角: {np.round(q_end, 4)}")

    assert sim_time >= SIM_TIME, f"仿真时间未推进到 {SIM_TIME} s: {sim_time:.2f}"
    assert tilt_end < 0.05, f"结束时姿态未回平: {tilt_end:.4f} rad"
    assert z_err < 0.2, f"结束时高度误差过大: {z_err:.3f} m"
    assert xy_drift < 0.5, f"结束时水平漂移过大: {xy_drift:.3f} m"
    assert np.allclose(sweep, 2.0, atol=0.2), (
        f"每个关节实际摆幅峰峰值应约 2 rad（±1 rad 摆动），实际 {sweep}"
    )
    assert np.allclose(q_end, 0.0, atol=0.1), f"淡出后关节应回到 0 位，实际 {q_end}"
    assert not sim.is_alive(), "仿真线程未退出"
    assert not drone_ctrl.is_alive(), "多旋翼控制器线程未退出"
    assert not arm_ctrl.is_alive(), "机械臂控制器线程未退出"

    print("\nPASS: 机械臂各关节 ±1 rad 摆动期间多旋翼全程保持平稳")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="UAMSim 综合演示脚本")
    parser.add_argument("--full", action="store_true",
                        help="完整集成演示（级联 PID + 机械臂摆动）")
    parser.add_argument("--headless", action="store_true",
                        help="无头模式（不开 viewer，5x 加速）")
    args = parser.parse_args()

    if args.full:
        run_full_demo(headless=args.headless)
    else:
        run_hover_simple(headless=args.headless)


if __name__ == "__main__":
    main()
