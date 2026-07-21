"""多旋翼控制器基类（独立控制线程，用户自定义控制器的父类）。

本类是一个**线程化的模板类**。三线程模型下，控制器**不注入仿真**，
而是在仿真编译完成后由外部（如 main.py）构造并 ``start()``：

1. **装配**（构造时，物理尚未启动，读视图安全）：自省旋翼几何、整机
   质量、推力范围并构建 4xN 控制分配矩阵伪逆（``_alloc_pinv`` / ``_mass``
   / ``_gravity`` 等受保护成员供子类直接使用）；
2. **执行器实例化**：按 rotor1..N 自动实例化
   :class:`~src.Actuators.RotorActuator`，注册进 ``channel``
   （``self.actuators`` 列表）；
3. **控制线程**：``run()`` 循环经帧同步邮箱接收仿真线程发布的
   ``SensorData`` 快照（存入 ``self.sensor``），调用 ``control()``，
   算完回执——控制频率与物理帧严格同步（1 kHz）。

**用户自定义控制器**：继承本类，只重写 ``control()`` 一个方法——
读取 ``self.sensor`` 快照，把推力写入 ``self.actuators[i].set_thrust()``。
指令只进执行器缓冲，由仿真线程在帧边界统一写入 mjData
（全进程只有仿真线程访问 mjData，无数据竞争）。

**稳定性保证（语法级）**：除 ``control`` 外的全部基类方法已被冻结——
子类定义同名方法时类创建即抛 ``TypeError``，从语法上杜绝仿真协议被
意外改写。
"""

from __future__ import annotations

import threading
import traceback

import numpy as np

from .Actuators import RotorActuator
from .FrameSync import ControllerChannel
from .Messages import SensorData


def _wrap_angle(angle: float) -> float:
    """角度卷绕到 [-pi, pi]。"""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class MultirotorController(threading.Thread):
    """多旋翼控制器线程基类（默认：位置/姿态串级 PID）。

    构造参数除 ``channel``（帧同步通道）与 ``multirotor``（平台视图，
    均已编译就绪）外均为标量或长度 3 向量（逐轴增益），仅作用于默认
    PID 实现；子类整体重写 ``control()`` 时可忽略这些参数。
    """

    # ---------- 子类化规则（语法级冻结） ----------
    # 用户自定义控制器只允许重写 _OVERRIDABLE_METHOD 一个方法；
    # 子类定义 _FROZEN_METHODS 中的同名方法时，类创建即抛 TypeError。
    _OVERRIDABLE_METHOD = "control"
    _FROZEN_METHODS = frozenset({
        "__init__",
        "__init_subclass__",
        "_OVERRIDABLE_METHOD",
        "_FROZEN_METHODS",
        "run",
        "set_target_position",
    })

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        illegal = sorted(set(vars(cls)) & MultirotorController._FROZEN_METHODS)
        if illegal:
            raise TypeError(
                f"{cls.__name__}: 为保证仿真链路稳定，控制器基类方法 {illegal} "
                "已冻结，不允许重写；用户自定义控制器只能重写 "
                f"'{MultirotorController._OVERRIDABLE_METHOD}' 一个方法"
            )

    def __init__(
        self,
        channel: ControllerChannel,
        multirotor,
        targetPosition=None,
        targetYaw: float = 0.0,
        positionKp=(1.5, 1.5, 4.0),
        positionKd=(1.2, 1.2, 2.5),
        positionKi=(0.15, 0.15, 0.6),
        attitudeKp=(6.0, 6.0, 1.5),
        attitudeKd=(0.35, 0.35, 0.15),
        attitudeKi=(0.4, 0.4, 0.0),
        integralLimit: float = 1.0,
        maxTilt: float = 0.6,
    ):
        super().__init__(daemon=True, name="MultirotorController")

        self._channel = channel
        self._target_position = (
            None if targetPosition is None else np.asarray(targetPosition, dtype=float).reshape(3)
        )
        self._target_yaw = float(targetYaw)

        self._kp_p = np.asarray(positionKp, dtype=float).reshape(3)
        self._kd_p = np.asarray(positionKd, dtype=float).reshape(3)
        self._ki_p = np.asarray(positionKi, dtype=float).reshape(3)
        self._kp_a = np.asarray(attitudeKp, dtype=float).reshape(3)
        self._kd_a = np.asarray(attitudeKd, dtype=float).reshape(3)
        self._ki_a = np.asarray(attitudeKi, dtype=float).reshape(3)
        self._integral_limit = float(integralLimit)
        self._max_tilt = float(maxTilt)

        # ---- 装配（构造时物理尚未启动，读 mjData 安全）----
        positions, yaw_coeffs, ctrl_ranges = multirotor.get_rotor_geometry()
        n = multirotor.n_rotors

        # 分配矩阵 A (4xN)：w = [T, τx, τy, τz] = A @ u
        # 第 i 列：推力 f_i 沿机体 +Z，作用于机体系 r_i 处 ->
        # τ = r_i × (f_i ẑ) + κ_i f_i ẑ，r × ẑ = [r_y, -r_x, 0]
        alloc = np.vstack([
            np.ones(n),                 # T
            positions[:, 1],            # τx =  r_y * f
            -positions[:, 0],           # τy = -r_x * f
            yaw_coeffs,                 # τz =  κ   * f
        ])
        self._alloc_pinv = np.linalg.pinv(alloc)
        self._n_rotors = n
        self._mass = float(multirotor.model.body_mass.sum())
        gravity = np.asarray(multirotor.model.opt.gravity, dtype=float)
        self._gravity = float(-gravity[2]) if gravity[2] < 0 else 9.81
        self._ctrl_min = ctrl_ranges[:, 0]
        self._ctrl_max = ctrl_ranges[:, 1]

        # ---- 自动实例化旋翼执行器并注册进通道（初始指令 = 均分悬停推力） ----
        hover = self._mass * self._gravity / n
        self.actuators: list[RotorActuator] = [
            RotorActuator(
                multirotor.data,
                multirotor.rotors[name],
                ctrl_ranges[i],
                hover,
                name=name,
            )
            for i, name in enumerate(multirotor.rotor_names)
        ]
        channel.attach(self.actuators)

        # ---- 运行状态 ----
        self.sensor: SensorData = multirotor.get_state()  # 最新快照（构造时取初始值）
        self._pos_integral = np.zeros(3)
        self._att_integral = np.zeros(3)

    # ---------- 目标注入 ----------

    def set_target_position(self, position, yaw: float | None = None) -> None:
        """设置位置目标（世界系 [x y z]），可选同时设置偏航目标 [rad]。"""
        self._target_position = np.asarray(position, dtype=float).reshape(3)
        if yaw is not None:
            self._target_yaw = float(yaw)

    # ---------- 线程入口（已冻结） ----------

    def run(self) -> None:
        """控制循环：等快照 -> control() -> 回执；邮箱关闭即退出。

        ``control()`` 抛异常不杀线程：打印堆栈（前 3 次）并沿用执行器
        缓冲中的上一帧指令。
        """
        mailbox = self._channel.mailbox
        last_frame = -1
        failures = 0
        while True:
            snapshot, frame = mailbox.wait_snapshot(last_frame)
            if snapshot is None:
                break
            last_frame = frame
            self.sensor = snapshot
            try:
                self.control()
            except Exception:
                failures += 1
                if failures <= 3:
                    print(f"[MultirotorController] control() 第 {failures} 次异常：")
                    traceback.print_exc()
            mailbox.mark_done(frame)

    # ---------- 控制律（用户重写点） ----------

    def control(self) -> None:
        """由 ``self.sensor`` 快照计算各旋翼推力并写入 ``self.actuators``。

        **用户自定义控制器只需重写本方法。** 默认实现为串级 PID：

        - 外环：位置/速度 PID -> 期望加速度（世界系）
        - 期望姿态：小角近似把水平加速度分配为 roll/pitch，叠加偏航目标
        - 总推力：期望加速度投影到机体 Z 轴
        - 内环：姿态 PID -> 期望力矩，经分配矩阵伪逆解出各旋翼推力
        - 推力截断由 :class:`RotorActuator` 在写入时自动完成
        """
        sensor = self.sensor
        dt = max(float(sensor.timestep), 1e-6)
        p = np.asarray(sensor.dronePosition, dtype=float)
        v = np.asarray(sensor.droneVelocity, dtype=float)
        euler = np.asarray(sensor.droneOrientation, dtype=float)
        omega = np.asarray(sensor.droneAngularVelocity, dtype=float)

        target = self._target_position if self._target_position is not None else p

        # ---- 外环：位置 PID -> 期望加速度（世界系） ----
        e_p = target - p
        self._pos_integral = np.clip(
            self._pos_integral + e_p * dt, -self._integral_limit, self._integral_limit
        )
        accel = self._kp_p * e_p - self._kd_p * v + self._ki_p * self._pos_integral

        # ---- 期望姿态（小角近似，目标偏航下分配 roll/pitch） ----
        psi = self._target_yaw
        roll_des = (accel[0] * np.sin(psi) - accel[1] * np.cos(psi)) / self._gravity
        pitch_des = (accel[0] * np.cos(psi) + accel[1] * np.sin(psi)) / self._gravity
        roll_des = float(np.clip(roll_des, -self._max_tilt, self._max_tilt))
        pitch_des = float(np.clip(pitch_des, -self._max_tilt, self._max_tilt))
        att_des = np.array([roll_des, pitch_des, psi])

        # ---- 总推力：期望加速度投影到机体 Z 轴（R @ ẑ 第三列） ----
        cr, sr = np.cos(euler[0]), np.sin(euler[0])
        cp, sp = np.cos(euler[1]), np.sin(euler[1])
        cy, sy = np.cos(euler[2]), np.sin(euler[2])
        body_z_world = np.array([
            cy * sp * cr + sy * sr,
            sy * sp * cr - cy * sr,
            cr * cp,
        ])
        thrust = self._mass * float(
            np.dot([accel[0], accel[1], self._gravity + accel[2]], body_z_world)
        )

        # ---- 内环：姿态 PID -> 期望力矩 ----
        e_att = att_des - euler
        e_att[2] = _wrap_angle(e_att[2])
        self._att_integral = np.clip(
            self._att_integral + e_att * dt, -self._integral_limit, self._integral_limit
        )
        torque = self._kp_a * e_att - self._kd_a * omega + self._ki_a * self._att_integral

        # ---- 混控：伪逆分配，写入执行器（截断由 RotorActuator 完成） ----
        wrench = np.array([thrust, torque[0], torque[1], torque[2]])
        u = self._alloc_pinv @ wrench
        for actuator, value in zip(self.actuators, u):
            actuator.set_thrust(float(value))
