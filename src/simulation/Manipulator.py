"""机械臂组件。

两阶段生命周期（与 Multirotor 同构）：

1. **读取 MJCF**（``__init__``）：``MjSpec.from_file`` 加载机械臂 MJCF，
   立即从 spec 自省关节、舵机 actuator、末端 site 与 sensor 等约定信息——
   此时模型尚未编译，不依赖任何 MjModel/MjData；
2. **绑定编译产物**（``bind``）：由 ``Environment.compile()`` 在统一编译后
   回调，把带命名空间前缀的名称解析为共享 MjModel/MjData 中的 id 与地址，
   此后 ``set_actuator`` / ``get_sensor`` 等读写接口可用。
"""

from __future__ import annotations

import pathlib

import numpy as np

try:
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None

from .. import _MODELS_DIR
from ..utils.PerceptionBus import SensorSnapshot


class Manipulator:
    """机械臂组件（读取自身 MJCF，编译后绑定共享模型）。

    MJCF 命名约定（``__init__`` 读取时校验）：

    - 关节：全部非 free 关节（``joint1 .. jointN``，按运动树顺序）
    - 舵机：每个关节各有一个 actuator（bind 时经 ``actuator_trnid`` 反查，
      不依赖命名）
    - 末端 site：``ee_site``；传感器：全部 sensor（如 ``jointposN`` /
      ``jointvelN`` / ``ee_pos`` / ``ee_quat``）
    """

    EE_SITE_NAME = "ee_site"

    def __init__(self, manipulatorPath: str | None = None):
        """读取机械臂 MJCF，自省关节 / actuator / sensor 信息（不编译）。"""
        if mujoco is None:
            raise ImportError("Manipulator 需要 mujoco 包，请先 pip install mujoco")

        path = pathlib.Path(manipulatorPath) if manipulatorPath else _MODELS_DIR / "Manipulator.xml"
        self.spec = mujoco.MjSpec.from_file(str(path))

        # ---- 关节自省（非 free，spec 文档序即运动树序） ----
        self._joint_locals = [
            j.name for j in self.spec.joints
            if int(j.type) != int(mujoco.mjtJoint.mjJNT_FREE)
        ]
        if not self._joint_locals:
            raise RuntimeError(f"Manipulator: '{path}' 中未找到任何非 free 关节")

        # ---- 舵机 actuator / 末端 site / sensor 自省 ----
        self._servo_locals = [a.name for a in self.spec.actuators]
        if not any(el.name == self.EE_SITE_NAME for el in self.spec.sites):
            raise RuntimeError(f"Manipulator: MJCF 中缺少末端 site '{self.EE_SITE_NAME}'")
        self._sensor_locals = [s.name for s in self.spec.sensors]

        # ---- 绑定产物（bind() 后填充） ----
        self._model = None
        self._data = None
        self._ns = ""
        self.joints: dict[str, int] = {}            # 全名 -> joint id
        self.servos: dict[str, int] = {}            # 关节全名 -> actuator id
        self._sensors: dict[str, tuple[int, int]] = {}  # 本地名 -> (adr, dim)
        self._joint_order: list[str] = []           # 关节全名，按运动树顺序
        self._qpos_adrs: list[int] = []
        self._qvel_adrs: list[int] = []
        self._ee_site_id = -1

    # ---------- 绑定（编译后由组合器回调） ----------

    def bind(self, model, data, namespace: str = "arm/") -> None:
        """绑定到统一编译出的共享 MjModel/MjData，解析全部 id 与地址。

        ``namespace`` 为组合时的名称前缀（场景内为 ``"uam/arm/"``）。
        """
        self._model = model
        self._data = data
        self._ns = namespace

        # ---- 关节：按编译后 joint id（运动树顺序）排序 ----
        self.joints = {}
        for local in self._joint_locals:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, namespace + local)
            if joint_id < 0:
                raise RuntimeError(f"Manipulator: 编译产物中缺少关节 '{namespace}{local}'")
            self.joints[namespace + local] = joint_id
        self._joint_order = sorted(self.joints, key=lambda n: self.joints[n])
        self._qpos_adrs = [int(model.jnt_qposadr[self.joints[n]]) for n in self._joint_order]
        self._qvel_adrs = [int(model.jnt_dofadr[self.joints[n]]) for n in self._joint_order]

        # ---- 舵机：actuator_trnid[act, 0] == joint_id 即绑定该关节 ----
        self.servos = {}
        for act_id in range(model.nu):
            if int(model.actuator_trntype[act_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            target_joint = int(model.actuator_trnid[act_id, 0])
            for name, joint_id in self.joints.items():
                if target_joint == joint_id:
                    self.servos[name] = act_id
        if len(self.servos) != len(self.joints):
            missing = sorted(set(self.joints) - set(self.servos))
            raise RuntimeError(f"Manipulator: 以下关节缺少对应舵机 actuator: {missing}")

        # ---- 末端 site ----
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, namespace + self.EE_SITE_NAME)
        if ee_id < 0:
            raise RuntimeError(f"Manipulator: 编译产物中缺少末端 site '{namespace}{self.EE_SITE_NAME}'")
        self._ee_site_id = ee_id

        # ---- 传感器 ----
        self._sensors = {}
        for local in self._sensor_locals:
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, namespace + local)
            if sensor_id >= 0:
                self._sensors[local] = (int(model.sensor_adr[sensor_id]), int(model.sensor_dim[sensor_id]))

    def _require_bound(self) -> None:
        if self._model is None or self._data is None:
            raise RuntimeError(
                "Manipulator 尚未绑定编译产物：请经 Environment.attach_robot() "
                "挂载编译后再调用"
            )

    # ---------- 属性 ----------

    @property
    def model(self):
        """绑定的共享 MjModel（只读句柄，供控制器做 Jacobian 逆解）。"""
        return self._model

    @property
    def data(self):
        """绑定的共享 MjData（只读句柄，供控制器做 Jacobian 逆解）。"""
        return self._data

    @property
    def n_joints(self) -> int:
        return len(self._joint_locals)

    @property
    def ee_site_id(self) -> int:
        """末端 site id（供 mj_jac 等底层调用）。"""
        self._require_bound()
        return self._ee_site_id

    @property
    def joint_names(self) -> list[str]:
        """关节全名列表（bind 后带命名空间前缀），按运动树顺序。"""
        if self._joint_order:
            return list(self._joint_order)
        return [self._ns + local for local in self._joint_locals]

    @property
    def sensor_names(self) -> list[str]:
        """MJCF 中声明的传感器本地名列表。"""
        return sorted(self._sensor_locals)

    # ---------- 控制写入 ----------

    def set_actuator(self, q) -> None:
        """写入各关节舵机 actuator 的目标角 [rad]（位置伺服）。"""
        self._require_bound()
        values = np.asarray(q, dtype=float).reshape(-1)
        if values.size != self.n_joints:
            raise ValueError(
                f"Manipulator.set_actuator: 期望 {self.n_joints} 个目标角，收到 {values.size} 个"
            )
        for name, value in zip(self._joint_order, values):
            self._data.ctrl[self.servos[name]] = float(value)

    # ---------- 状态读取 ----------

    def get_joint_positions(self) -> np.ndarray:
        self._require_bound()
        return np.array([self._data.qpos[adr] for adr in self._qpos_adrs], dtype=float)

    def get_joint_velocities(self) -> np.ndarray:
        self._require_bound()
        return np.array([self._data.qvel[adr] for adr in self._qvel_adrs], dtype=float)

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """末端执行器的世界位姿（位置 + 3x3 旋转矩阵）。"""
        self._require_bound()
        pos = self._data.site_xpos[self._ee_site_id].copy()
        mat = self._data.site_xmat[self._ee_site_id].reshape(3, 3).copy()
        return pos, mat

    def get_sensor(self, local_key: str) -> np.ndarray:
        """按本地名读取传感器（如 'jointpos1' / 'ee_pos'）。"""
        self._require_bound()
        if local_key not in self._sensors:
            raise KeyError(
                f"Manipulator: 传感器 '{self._ns}{local_key}' 不存在（已发现: {sorted(self._sensors)}）"
            )
        adr, dim = self._sensors[local_key]
        return np.array(self._data.sensordata[adr: adr + dim], dtype=float)

    def get_state(self) -> SensorSnapshot:
        """提取机械臂传感快照（供帧同步发布给控制器线程）。

        含末端位置雅可比在本臂各关节 dof 上的切片 (3, n_joints)——
        雅可比依赖 mjData，只能由仿真线程（或主循环）求取。
        """
        self._require_bound()
        ee_pos = self._data.site_xpos[self._ee_site_id]
        body_id = int(self._model.site_bodyid[self._ee_site_id])
        jacp = np.zeros((3, self._model.nv))
        mujoco.mj_jac(self._model, self._data, jacp, None, ee_pos.copy(), body_id)
        return SensorSnapshot(
            timestamp=float(self._data.time),
            timestep=float(self._model.opt.timestep),
            jointPositions=self.get_joint_positions(),
            jointVelocities=self.get_joint_velocities(),
            eePosition=ee_pos.copy(),
            eeJacobian=jacp[:, self._qvel_adrs].copy(),
        )
