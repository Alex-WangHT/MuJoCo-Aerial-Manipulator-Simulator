"""整机集成测试：级联 PID 多旋翼控制 + 机械臂各关节 ±1 rad 摆动（带 GUI）。

直接运行（默认打开 MuJoCo viewer，实时 1x）：
    .venv/Scripts/python tests/test_integrated.py
无头模式（5x 加速，无画面）：
    .venv/Scripts/python tests/test_integrated.py --headless

场景：
- 多旋翼控制器（MultirotorController 子类）：级联 PID——外环位置 PID 出期望
  加速度并解算期望姿态，内环姿态 PID 出期望力矩，最后经混控矩阵（由
  get_rotor_geometry() 自省的旋翼几何构造）分配到各旋翼推力；
- 机械臂控制器（ManipulatorController 子类）：每个关节以 1 rad 幅值正弦
  摆动（周期 4 s，1 s 淡入/淡出包络保证零速起停）。摆动中心按关节范围
  选取：joint2 上限 0.174 / joint3 下限 -0.174，无法绕 0 对称摆 ±1 rad，
  故 joint1 绕 0、joint2 绕 -1、joint3 绕 +1，每关节峰峰值 2 rad；
- 控制目标：机械臂摆动全程多旋翼保持平稳。

平稳性由 TelemetryBuffer（仿真线程 1 kHz 写入，线程安全）**全程连续监控**，
任一时刻超限立即失败，而非只看结尾：

1. 全程任意时刻滚转/俯仰角 < 0.20 rad（~11 deg）、高度误差 < 0.3 m、
   水平漂移 < 0.8 m（fail-fast）；
2. 摆动停止并稳定后：姿态回平（< 0.05 rad）、高度误差 < 0.2 m、
   水平漂移 < 0.5 m；
3. 每个关节实际摆幅峰峰值 ≈ 2 rad（±0.2，由控制器逐帧记录实际关节角极值）；
4. 结束后仿真线程与两个控制器线程均干净退出。

GUI 模式下提前关闭 viewer 视为测试失败（仿真时间未跑满）。
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src import (
    Environment,
    Manipulator,
    ManipulatorController,
    MujocoSimulation,
    Multirotor,
    MultirotorController,
    TelemetryBuffer,
)

SPAWN_Z = 1.5          # 出生高度 [m]（environment.xml 的 uam_spawn）
SWING_PERIOD = 4.0     # 关节正弦摆动周期 [s]
SWING_STOP = 7.0       # 摆动淡出结束时刻 [s]（6~7 s 淡出，之后保持 0 位）
SIM_TIME = 9.0         # 总仿真时长 [s]（停摆后留 2 s 稳定）
TILT_LIMIT = 0.20      # 全程任意时刻最大允许滚转/俯仰角 [rad]
Z_ERR_LIMIT = 0.3      # 全程任意时刻最大允许高度误差 [m]
XY_LIMIT = 0.8         # 全程任意时刻最大允许水平漂移 [m]


class CascadedPidMultirotor(MultirotorController):
    """级联 PID 多旋翼控制器。

    外环（位置）：位置 PID -> 期望加速度 -> 期望总推力 + 期望滚转/俯仰；
    内环（姿态）：姿态 PID（含积分，消除机械臂重力矩引起的稳态倾角）
    -> 期望三轴力矩；混控矩阵 mixer（N x 4，伪逆预分配）把
    [总推力, τx, τy, τz] 映射为各旋翼推力。
    """

    def controller(self, feedback, mass, gravity, pos_ref, gains, mixer, state):
        pos = feedback.dronePosition
        vel = feedback.droneVelocity
        att = feedback.droneOrientation          # roll / pitch / yaw
        omega = feedback.droneAngularVelocity    # 机体系角速度
        dt = max(feedback.timestep, 1e-6)

        # ---- 外环：位置 PID -> 期望加速度 ----
        err = pos_ref - pos
        state["int_z"] = np.clip(
            state["int_z"] + err[2] * dt, -gains["int_z_limit"], gains["int_z_limit"]
        )
        ax = gains["kp_xy"] * err[0] - gains["kd_xy"] * vel[0]
        ay = gains["kp_xy"] * err[1] - gains["kd_xy"] * vel[1]
        az = (gains["kp_z"] * err[2] - gains["kd_z"] * vel[2]
              + gains["ki_z"] * state["int_z"])

        thrust = mass * (gravity + az)           # 期望总推力（小角度近似）

        # ---- 期望姿态：水平加速度解算为滚转/俯仰（yaw 补偿）----
        psi = att[2]
        roll_des = np.clip((np.sin(psi) * ax - np.cos(psi) * ay) / gravity, -0.3, 0.3)
        pitch_des = np.clip((np.cos(psi) * ax + np.sin(psi) * ay) / gravity, -0.3, 0.3)
        att_des = np.array([roll_des, pitch_des, 0.0])

        # ---- 内环：姿态 PID（积分消除稳态倾角）-> 期望力矩 ----
        att_err = att_des - att
        state["int_att"] = np.clip(
            state["int_att"] + att_err * dt, -gains["int_att_limit"], gains["int_att_limit"]
        )
        tau = (gains["kp_att"] * att_err - gains["kd_att"] * omega
               + gains["ki_att"] * state["int_att"])

        # ---- 混控：[总推力, τx, τy, τz] -> 各旋翼推力 ----
        return mixer @ np.array([thrust, tau[0], tau[1], tau[2]])


class SwingJointController(ManipulatorController):
    """机械臂各关节 ±1 rad 正弦摆动：q = envelope(t) · (center + sin(2π t / period))。

    - ``center``：摆动中心，按关节范围选取（joint2 上限 0.174 / joint3 下限
      -0.174，无法绕 0 对称摆 ±1 rad）：joint1 绕 0、joint2 绕 -1、
      joint3 绕 +1，每关节峰峰值 2 rad；
    - ``envelope``：1 s 淡入 + 停摆前 1 s 淡出的 smoothstep 包络，保证零速
      起停，淡出后各关节回到 0 位；
    - 逐帧在 ``state`` 里记录实际关节角历史极值，供测试校验实际摆幅。
    """

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = min(max(x, 0.0), 1.0)
        return x * x * (3.0 - 2.0 * x)

    def controller(self, feedback, center, period, swing_stop, state):
        np.minimum(state["q_min"], feedback.jointPositions, out=state["q_min"])
        np.maximum(state["q_max"], feedback.jointPositions, out=state["q_max"])
        t = feedback.timestamp
        envelope = self._smoothstep(t)
        if swing_stop is not None:               # None = 不停摆（持续观察模式）
            envelope = min(envelope, self._smoothstep(swing_stop - t))
        return envelope * (center + np.sin(2.0 * np.pi * t / period))


def build_mixer(multirotor) -> np.ndarray:
    """由旋翼几何构造混控伪逆矩阵 (N, 4)：[F, τx, τy, τz] -> u。"""
    positions, yaw_coeffs, _ = multirotor.get_rotor_geometry()
    alloc = np.vstack([
        np.ones(multirotor.n_rotors),    # 总推力
        positions[:, 1],                 # τx = Σ y_i · u_i
        -positions[:, 0],                # τy = Σ -x_i · u_i
        yaw_coeffs,                      # τz = Σ gear_z_i · u_i
    ])
    return alloc.T @ np.linalg.inv(alloc @ alloc.T)


def main() -> None:
    parser = argparse.ArgumentParser(description="整机集成测试（级联 PID + 各关节 1 rad）")
    parser.add_argument("--headless", action="store_true",
                        help="无头模式：不开 viewer，5x 加速跑完")
    args = parser.parse_args()
    use_viewer = not args.headless

    print("=" * 64)
    print("整机集成测试：级联 PID 多旋翼 + 机械臂各关节 ±1 rad 摆动")
    print(f"模式: {'GUI（实时 1x，viewer 已打开）' if use_viewer else '无头（5x 加速）'}")
    print("=" * 64)

    shutdown = threading.Event()
    telemetry = TelemetryBuffer(maxSamples=20000)   # 1 kHz x 20 s 足够覆盖
    env = Environment()
    sim = MujocoSimulation(shutdown, env, Multirotor(), Manipulator(),
                           telemetryBuffer=telemetry, useViewer=use_viewer,
                           realTimeFactor=1.0 if use_viewer else 5.0)
    sim.start()
    assert sim.wait_ready(timeout=10.0), "场景编译超时"
    uam = sim.uam

    # ---- 多旋翼：级联 PID ----
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

    # ---- 机械臂：各关节 ±1 rad 正弦摆动（中心按关节范围选取）----
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

    # ---- 全程连续监控（遥测线程安全，不触碰 mjData）：任一时刻超限立即失败 ----
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
            assert monitor["max_tilt"] < TILT_LIMIT, (
                f"t={sim_time:.2f}s 倾角 {monitor['max_tilt']:.4f} rad 超限 {TILT_LIMIT} rad"
            )
            assert monitor["max_z_err"] < Z_ERR_LIMIT, (
                f"t={sim_time:.2f}s 高度误差 {monitor['max_z_err']:.3f} m 超限 {Z_ERR_LIMIT} m"
            )
            assert monitor["max_xy"] < XY_LIMIT, (
                f"t={sim_time:.2f}s 水平漂移 {monitor['max_xy']:.3f} m 超限 {XY_LIMIT} m"
            )
        time.sleep(0.1)

    if sim_time < SIM_TIME and not sim.is_alive():
        shutdown.set()
        sim.join(timeout=5.0)
        raise AssertionError(
            f"仿真提前结束（viewer 被关闭？）：仿真时间 {sim_time:.2f} s < {SIM_TIME} s"
        )

    shutdown.set()
    sim.join(timeout=5.0)
    drone_ctrl.join(timeout=2.0)
    arm_ctrl.join(timeout=2.0)

    # ---- 线程已全部退出，此后读 mjData 安全 ----
    final_state = uam.multirotor.get_state()
    q_end = uam.manipulator.get_joint_positions()
    sweep = arm_ctrl.params["state"]["q_max"] - arm_ctrl.params["state"]["q_min"]
    tilt_end = max(abs(final_state.droneOrientation[0]), abs(final_state.droneOrientation[1]))
    z_err = abs(final_state.dronePosition[2] - SPAWN_Z)
    xy_drift = float(np.linalg.norm(final_state.dronePosition[:2]))

    print(f"仿真时间: {sim_time:.2f} s")
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

    print("PASS: 机械臂各关节 ±1 rad 摆动期间多旋翼全程保持平稳")


if __name__ == "__main__":
    main()
