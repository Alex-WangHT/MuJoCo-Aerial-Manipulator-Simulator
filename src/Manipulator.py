"""机械臂视图。

不持有独立物理模型，绑定到组合后编译出的共享 MjModel/MjData，
按约定名称（命名空间 + 本地名）从 MJCF 自省关节、舵机与末端执行器。
"""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None


class Manipulator:
    """机械臂视图。

    绑定到组合后编译出的共享 ``MjModel``/``MjData``，只管理带命名空间前缀
    （默认 ``arm/``）的那部分元素，全部从 MJCF 自省：

    - 关节：扫描所有 joint，取名称带前缀且非 free 类型者，按运动树顺序排列
    - 舵机：通过 ``actuator_trnid`` 反查绑定到各关节的 actuator（不依赖命名）
    - 末端：``ee_site``；传感器：所有带前缀的 sensor（adr + dim）
    """

    EE_SITE_NAME = "ee_site"

    def __init__(self, model, data, namespace: str = "arm/"):
        if mujoco is None:
            raise ImportError("Manipulator 需要 mujoco 包，请先 pip install mujoco")
        self._model = model
        self._data = data
        self._ns = namespace

        # ---- 关节自省（前缀 + 非 free，按 joint id 即运动树顺序） ----
        self.joints: dict[str, int] = {}
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
            if not name.startswith(namespace):
                continue
            if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            self.joints[name] = joint_id
        if not self.joints:
            raise RuntimeError(
                f"Manipulator: 在命名空间 '{namespace}' 下未找到任何关节，"
                "请检查 Manipulator.xml 或 attach 前缀"
            )
        self._joint_order = sorted(self.joints, key=lambda n: self.joints[n])
        self._qpos_adrs = [int(model.jnt_qposadr[self.joints[n]]) for n in self._joint_order]
        self._qvel_adrs = [int(model.jnt_dofadr[self.joints[n]]) for n in self._joint_order]

        # ---- 舵机自省：actuator_trnid[act, 0] == joint_id 即绑定该关节 ----
        self.servos: dict[str, int] = {}
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
            raise RuntimeError(f"Manipulator: MJCF 中缺少末端 site '{namespace}{self.EE_SITE_NAME}'")
        self._ee_site_id = ee_id

        # ---- 传感器自省（所有带前缀的 sensor） ----
        self._sensors: dict[str, tuple[int, int]] = {}
        for sensor_id in range(model.nsensor):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id) or ""
            if name.startswith(namespace):
                local = name[len(namespace):]
                self._sensors[local] = (int(model.sensor_adr[sensor_id]), int(model.sensor_dim[sensor_id]))

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
        return len(self.joints)

    @property
    def ee_site_id(self) -> int:
        """末端 site id（供 mj_jac 等底层调用）。"""
        return self._ee_site_id

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_order)

    @property
    def sensor_names(self) -> list[str]:
        return sorted(self._sensors)

    # ---------- 控制写入 ----------

    def set_joint_targets(self, q) -> None:
        """写入各关节舵机的目标角 [rad]（位置伺服）。"""
        values = np.asarray(q, dtype=float).reshape(-1)
        if values.size != self.n_joints:
            raise ValueError(
                f"Manipulator.set_joint_targets: 期望 {self.n_joints} 个目标角，收到 {values.size} 个"
            )
        for name, value in zip(self._joint_order, values):
            self._data.ctrl[self.servos[name]] = float(value)

    # ---------- 状态读取 ----------

    def get_joint_positions(self) -> np.ndarray:
        return np.array([self._data.qpos[adr] for adr in self._qpos_adrs], dtype=float)

    def get_joint_velocities(self) -> np.ndarray:
        return np.array([self._data.qvel[adr] for adr in self._qvel_adrs], dtype=float)

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """末端执行器的世界位姿（位置 + 3x3 旋转矩阵）。"""
        pos = self._data.site_xpos[self._ee_site_id].copy()
        mat = self._data.site_xmat[self._ee_site_id].reshape(3, 3).copy()
        return pos, mat

    def get_sensor(self, local_key: str) -> np.ndarray:
        """按去掉前缀的本地名读取传感器（如 'jointpos1' / 'ee_pos'）。"""
        if local_key not in self._sensors:
            raise KeyError(
                f"Manipulator: 传感器 '{self._ns}{local_key}' 不存在（已发现: {sorted(self._sensors)}）"
            )
        adr, dim = self._sensors[local_key]
        return np.array(self._data.sensordata[adr: adr + dim], dtype=float)

