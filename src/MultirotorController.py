"""多旋翼控制器基类（用户自定义控制器的父类）。

本类做两件事：

1. **协议与装配**：实现 ``MujocoSimulation`` 的 ``droneController`` 协议
   （``setSensorData`` / ``getControlInput``），
   并在 ``bind(multirotor)`` 中备好控制律所需的全部模型信息——
   旋翼几何、整机质量、推力范围与 4xN 控制分配矩阵伪逆；
2. **默认控制律**：``compute_control()`` 给出串级 PID（位置环 -> 姿态环 ->
   混控）参考实现。

**用户自定义控制器**：继承本类，只重写 ``compute_control(sensor) -> u``
一个方法即可——协议、装配（bind）、混控矩阵与推力截断全部由基类完成。
基类备好的 ``_alloc_pinv`` / ``_mass`` / ``_gravity`` / ``_target_position``
等受保护成员可直接使用。

**稳定性保证（语法级）**：除 ``compute_control`` 外的全部基类方法
（``__init__`` / ``bind`` / 协议方法 / 目标注入）已被冻结——子类定义
同名方法时类创建即抛 ``TypeError``，从语法上杜绝仿真协议被意外改写。
"""

from __future__ import annotations

import numpy as np

from .Messages import ControlInput, SensorData


def _wrap_angle(angle: float) -> float:
    """角度卷绕到 [-pi, pi]。"""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class MultirotorController:
    """多旋翼控制器基类（默认：位置/姿态串级 PID）。

    参数均为标量或长度 3 向量（逐轴增益），仅作用于默认 PID 实现；
    子类整体重写 ``compute_control`` 时可忽略这些参数。
    """

    # ---------- 子类化规则（语法级冻结） ----------
    # 用户自定义控制器只允许重写 _OVERRIDABLE_METHOD 一个方法；
    # 子类定义 _FROZEN_METHODS 中的同名方法时，类创建即抛 TypeError。
    _OVERRIDABLE_METHOD = "compute_control"
    _FROZEN_METHODS = frozenset({
        "__init__",
        "__init_subclass__",
        "_OVERRIDABLE_METHOD",
        "_FROZEN_METHODS",
        "bind",
        "set_target_position",
        "setSensorData",
        "getControlInput",
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

        # bind() 后填充（子类可直接使用的模型信息）
        self._bound = False
        self._mass = 0.0                    # 整机质量 [kg]
        self._gravity = 9.81                # 重力加速度 [m/s^2]
        self._alloc_pinv: np.ndarray | None = None     # (N, 4) 分配矩阵伪逆
        self._ctrl_min: np.ndarray | None = None       # (N,) 推力下限 [N]
        self._ctrl_max: np.ndarray | None = None       # (N,) 推力上限 [N]
        self._n_rotors = 0

        # 运行状态
        self._pos_integral = np.zeros(3)
        self._att_integral = np.zeros(3)
        self._u: np.ndarray | None = None

    # ---------- 组装 ----------

    def bind(self, multirotor) -> None:
        """从平台视图读取旋翼几何、推力范围与整机质量，构建混控器。

        由 ``MujocoSimulation._build()`` 在模型编译完成后自动调用；
        手动使用时在 ``Environment.compile()`` / ``AerialManipulator.compile()``
        之后调用。子类重写时通常应先调用 ``super().bind(multirotor)``。
        """
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
        self._ctrl_min = ctrl_ranges[:, 0]
        self._ctrl_max = ctrl_ranges[:, 1]
        self._n_rotors = n
        self._mass = float(multirotor.model.body_mass.sum())
        gravity = np.asarray(multirotor.model.opt.gravity, dtype=float)
        self._gravity = float(-gravity[2]) if gravity[2] < 0 else 9.81

        self._u = np.full(n, self._mass * self._gravity / n)
        self._bound = True

    # ---------- 目标注入 ----------

    def set_target_position(self, position, yaw: float | None = None) -> None:
        """设置位置目标（世界系 [x y z]），可选同时设置偏航目标 [rad]。"""
        self._target_position = np.asarray(position, dtype=float).reshape(3)
        if yaw is not None:
            self._target_yaw = float(yaw)

    # ---------- droneController 协议（一般无需重写） ----------

    def setSensorData(self, sensor: SensorData) -> None:
        """接收一帧传感数据并同步计算控制输出（1 kHz 调用）。"""
        if not self._bound:
            raise RuntimeError(
                "MultirotorController 尚未 bind(multirotor)，"
                "请将其传给 MujocoSimulation(droneController=...) 由仿真自动绑定"
            )
        u = np.asarray(self.compute_control(sensor), dtype=float).reshape(-1)
        if u.size != self._n_rotors:
            raise ValueError(
                f"compute_control 返回了 {u.size} 个推力值，应为 {self._n_rotors} 个"
            )
        self._u = np.clip(u, self._ctrl_min, self._ctrl_max)

    def getControlInput(self) -> ControlInput:
        if self._u is None:
            raise RuntimeError("MultirotorController: setSensorData 尚未被调用")
        return ControlInput(u=self._u.copy())

    # ---------- 控制律（用户重写点） ----------

    def compute_control(self, sensor: SensorData) -> np.ndarray:
        """由传感数据计算各旋翼推力 [N]（返回未截断值，基类负责截断）。

        **用户自定义控制器只需重写本方法。** 默认实现为串级 PID：

        - 外环：位置/速度 PID -> 期望加速度（世界系）
        - 期望姿态：小角近似把水平加速度分配为 roll/pitch，叠加偏航目标
        - 总推力：期望加速度投影到机体 Z 轴
        - 内环：姿态 PID -> 期望力矩，经分配矩阵伪逆解出各旋翼推力
        """
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

        # ---- 混控：伪逆分配（截断由基类 setSensorData 统一处理） ----
        wrench = np.array([thrust, torque[0], torque[1], torque[2]])
        return self._alloc_pinv @ wrench
