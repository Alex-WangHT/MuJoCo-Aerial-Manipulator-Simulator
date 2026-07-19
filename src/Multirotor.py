"""多旋翼平台视图。

不持有独立物理模型，绑定到组合后编译出的共享 MjModel/MjData，
按约定名称（命名空间 + 本地名）从 MJCF 自省旋翼、机体状态、IMU 与挂载点。
"""

from __future__ import annotations

import re

import numpy as np

try:
    import mujoco
except Exception:  # pragma: no cover - 允许未安装 mujoco 时导入本模块
    mujoco = None

from .Messages import SensorData

from . import quaternionToEuler


class Multirotor:
    """多旋翼平台视图。

    不持有独立的物理模型，而是绑定到组合后编译出的共享 ``MjModel``/``MjData``
    上，按约定名称从 MJCF 自省：

    - 旋翼 actuator：扫描所有 actuator，匹配 ``rotorN``（驱动关节位置）
    - 机体/自由关节/IMU/挂载点：按固定名称查找
    - 传感器：按固定名称查找 ``model.sensor_adr``（传感器位置）
    """

    BODY_NAME = "drone"
    FREEJOINT_NAME = "drone_free"
    IMU_SITE_NAME = "imu_site"
    MOUNT_SITE_NAME = "manipulator_mount"

    _ROTOR_PATTERN = re.compile(r"^rotor(\d+)$")
    # 约定名称的可选传感器 -> SensorData 之外的读取入口
    _SENSOR_KEYS = ("gyro", "accel", "mag", "baro", "drone_pos", "drone_quat", "mount_pos", "mount_quat")

    def __init__(self, model, data, namespace: str = ""):
        if mujoco is None:
            raise ImportError("Multirotor 需要 mujoco 包，请先 pip install mujoco")
        self._model = model
        self._data = data
        self._ns = namespace

        # ---- 旋翼自省（按 rotorN 命名约定，按编号排序） ----
        self.rotors: dict[str, int] = {}
        for act_id in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id) or ""
            if not name.startswith(namespace):
                continue
            if self._ROTOR_PATTERN.match(name[len(namespace):]):
                self.rotors[name] = act_id
        if not self.rotors:
            raise RuntimeError(
                f"Multirotor: 在命名空间 '{namespace}' 下未找到任何 rotorN actuator，"
                "请检查 multirotor.xml 的 actuator 命名约定"
            )
        self._rotor_order = sorted(
            self.rotors,
            key=lambda n: int(self._ROTOR_PATTERN.match(n[len(namespace):]).group(1)),
        )

        # ---- 机体 / 自由关节 / site ----
        self._body_id = self._lookup(mujoco.mjtObj.mjOBJ_BODY, self.BODY_NAME)
        self._joint_id = self._lookup(mujoco.mjtObj.mjOBJ_JOINT, self.FREEJOINT_NAME)
        self._qpos_adr = int(model.jnt_qposadr[self._joint_id])
        self._qvel_adr = int(model.jnt_dofadr[self._joint_id])
        self._imu_site_id = self._lookup(mujoco.mjtObj.mjOBJ_SITE, self.IMU_SITE_NAME)
        self.mount_site_id = self._lookup(mujoco.mjtObj.mjOBJ_SITE, self.MOUNT_SITE_NAME)

        # ---- 传感器自省（adr + dim，可选传感器缺失时跳过） ----
        self._sensors: dict[str, tuple[int, int]] = {}
        for key in self._SENSOR_KEYS:
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, namespace + key)
            if sensor_id >= 0:
                self._sensors[key] = (int(model.sensor_adr[sensor_id]), int(model.sensor_dim[sensor_id]))

    # ---------- 名称查找 ----------

    def _lookup(self, obj_type, local_name: str) -> int:
        full_name = self._ns + local_name
        obj_id = mujoco.mj_name2id(self._model, obj_type, full_name)
        if obj_id < 0:
            raise RuntimeError(f"Multirotor: MJCF 中缺少必需的元素 '{full_name}'")
        return obj_id

    # ---------- 属性 ----------

    @property
    def n_rotors(self) -> int:
        return len(self.rotors)

    @property
    def rotor_names(self) -> list[str]:
        return list(self._rotor_order)

    @property
    def model(self):
        """绑定的共享 MjModel（只读句柄，供控制器读取几何/质量/gear）。"""
        return self._model

    @property
    def body_id(self) -> int:
        return self._body_id

    # ---------- 控制写入 ----------

    def set_thrusts(self, u) -> None:
        """按 rotor1..rotorN 顺序写入旋翼推力 [N]。"""
        values = np.asarray(u, dtype=float).reshape(-1)
        if values.size != self.n_rotors:
            raise ValueError(f"Multirotor.set_thrusts: 期望 {self.n_rotors} 个推力值，收到 {values.size} 个")
        for name, value in zip(self._rotor_order, values):
            self._data.ctrl[self.rotors[name]] = float(value)

    # ---------- 状态读取 ----------

    def get_state(self) -> SensorData:
        """从自由关节 qpos/qvel 提取平台位姿与速度。"""
        qpos = self._data.qpos[self._qpos_adr: self._qpos_adr + 7]
        qvel = self._data.qvel[self._qvel_adr: self._qvel_adr + 6]
        return SensorData(
            timestamp=float(self._data.time),
            timestep=float(self._model.opt.timestep),
            dronePosition=qpos[:3].copy(),
            droneVelocity=qvel[:3].copy(),
            droneOrientation=quaternionToEuler(qpos[3:7]),
            droneAngularVelocity=qvel[3:6].copy(),
        )

    def get_sensor(self, key: str) -> np.ndarray:
        """按约定名称读取传感器数据（如 'gyro' / 'accel' / 'drone_pos'）。"""
        if key not in self._sensors:
            raise KeyError(f"Multirotor: 传感器 '{self._ns}{key}' 不存在（已发现: {sorted(self._sensors)}）")
        adr, dim = self._sensors[key]
        return np.array(self._data.sensordata[adr: adr + dim], dtype=float)

    def get_imu(self) -> dict[str, np.ndarray]:
        """读取 IMU 数据：陀螺仪、加速度计（以及磁强计/气压计，若模型提供）。"""
        imu = {
            "gyro": self.get_sensor("gyro"),
            "accel": self.get_sensor("accel"),
        }
        for optional_key in ("mag", "baro"):
            if optional_key in self._sensors:
                imu[optional_key] = self.get_sensor(optional_key)
        return imu

    def get_mount_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """机械臂挂载点的世界位姿（位置 + 3x3 旋转矩阵）。"""
        pos = self._data.site_xpos[self.mount_site_id].copy()
        mat = self._data.site_xmat[self.mount_site_id].reshape(3, 3).copy()
        return pos, mat

    def get_rotor_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """旋翼几何自省（供控制分配/混控器使用）。

        返回 ``(positions, yaw_coeffs, ctrl_ranges)``：

        - ``positions``：``(N, 3)``，各旋翼 site 在**机体系**下的坐标
          （由世界坐标经机体旋转矩阵反变换得到，与机体姿态无关）
        - ``yaw_coeffs``：``(N,)``，各旋翼单位推力产生的 Z 反扭矩系数
          （取自 ``actuator_gear[:, 5]``，正/反转桨符号交替）
        - ``ctrl_ranges``：``(N, 2)``，各旋翼推力上下限 [N]
        """
        drone_pos = self._data.xpos[self._body_id]
        drone_mat = self._data.xmat[self._body_id].reshape(3, 3)
        positions = np.zeros((self.n_rotors, 3))
        yaw_coeffs = np.zeros(self.n_rotors)
        ctrl_ranges = np.zeros((self.n_rotors, 2))
        for i, name in enumerate(self._rotor_order):
            act_id = self.rotors[name]
            site_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_SITE, name + "_site"
            )
            if site_id < 0:
                raise RuntimeError(f"Multirotor: MJCF 中缺少旋翼 site '{name}_site'")
            world_rel = self._data.site_xpos[site_id] - drone_pos
            positions[i] = drone_mat.T @ world_rel
            yaw_coeffs[i] = float(self._model.actuator_gear[act_id, 5])
            ctrl_ranges[i] = self._model.actuator_ctrlrange[act_id]
        return positions, yaw_coeffs, ctrl_ranges

