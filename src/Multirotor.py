"""多旋翼平台组件。

两阶段生命周期：

1. **读取 MJCF**（``__init__``）：``MjSpec.from_file`` 加载多旋翼 MJCF，
   立即从 spec 自省 actuator（``rotorN`` 命名约定）、sensor、机体/自由关节/
   site 等约定信息——此时模型尚未编译，不依赖任何 MjModel/MjData；
2. **绑定编译产物**（``bind``）：由 ``Environment.compile()`` 在统一编译后
   回调，把名称解析为共享 MjModel/MjData 中的 id 与地址，此后
   ``set_actuator`` / ``get_sensor`` 等读写接口可用。
"""

from __future__ import annotations

import pathlib
import re

import numpy as np

try:
    import mujoco
except Exception:  # pragma: no cover - 允许未安装 mujoco 时导入本模块
    mujoco = None

from . import _MODELS_DIR
from .Messages import SensorData

from . import quaternionToEuler


class Multirotor:
    """多旋翼平台组件（读取自身 MJCF，编译后绑定共享模型）。

    MJCF 命名约定（``__init__`` 读取时校验）：

    - 旋翼 actuator：``rotor1 .. rotorN``（site 传动，gear 含 Z 反扭矩）
    - 旋翼 site：``rotorN_site``；机体 body：``drone``；自由关节：``drone_free``
    - IMU site：``imu_site``；机械臂挂载点 site：``manipulator_mount``
    - 传感器：``gyro`` / ``accel`` / ``mag`` / ``drone_pos`` / ``drone_quat`` /
      ``mount_pos`` / ``mount_quat``（可选，缺什么跳过什么）
    """

    BODY_NAME = "drone"
    FREEJOINT_NAME = "drone_free"
    IMU_SITE_NAME = "imu_site"
    MOUNT_SITE_NAME = "manipulator_mount"

    _ROTOR_PATTERN = re.compile(r"^rotor(\d+)$")

    def __init__(self, multirotorPath: str | None = None):
        """读取多旋翼 MJCF，自省 actuator / sensor 等约定信息（不编译）。"""
        if mujoco is None:
            raise ImportError("Multirotor 需要 mujoco 包，请先 pip install mujoco")

        path = pathlib.Path(multirotorPath) if multirotorPath else _MODELS_DIR / "multirotor.xml"
        self.spec = mujoco.MjSpec.from_file(str(path))

        # ---- actuator 自省：rotorN 命名约定，按编号排序 ----
        self._rotor_locals = sorted(
            (a.name for a in self.spec.actuators if self._ROTOR_PATTERN.match(a.name or "")),
            key=lambda n: int(self._ROTOR_PATTERN.match(n).group(1)),
        )
        if not self._rotor_locals:
            raise RuntimeError(
                f"Multirotor: '{path}' 中未找到任何 rotorN actuator，"
                "请检查 MJCF 的 actuator 命名约定"
            )

        # ---- 机体 / 自由关节 / site 校验 ----
        self._require_spec_element(self.spec.bodies, self.BODY_NAME, "body")
        self._require_spec_element(self.spec.joints, self.FREEJOINT_NAME, "joint")
        self._require_spec_element(self.spec.sites, self.IMU_SITE_NAME, "site")
        self._require_spec_element(self.spec.sites, self.MOUNT_SITE_NAME, "site")
        for rotor in self._rotor_locals:
            self._require_spec_element(self.spec.sites, rotor + "_site", "site")

        # ---- sensor 自省（可选传感器缺失时跳过） ----
        self._sensor_locals = [s.name for s in self.spec.sensors]

        # ---- 绑定产物（bind() 后填充） ----
        self._model = None
        self._data = None
        self._ns = ""
        self.rotors: dict[str, int] = {}          # 全名 -> actuator id
        self._sensors: dict[str, tuple[int, int]] = {}  # 本地名 -> (adr, dim)
        self._body_id = -1
        self._qpos_adr = -1
        self._qvel_adr = -1
        self._imu_site_id = -1
        self.mount_site_id = -1

    @staticmethod
    def _require_spec_element(elements, name: str, kind: str) -> None:
        if not any(el.name == name for el in elements):
            raise RuntimeError(f"Multirotor: MJCF 中缺少必需的 {kind} '{name}'")

    # ---------- 绑定（编译后由组合器回调） ----------

    def bind(self, model, data, namespace: str = "") -> None:
        """绑定到统一编译出的共享 MjModel/MjData，解析全部 id 与地址。

        ``namespace`` 为组合时的名称前缀（场景内为 ``"uam/"``）。
        """
        self._model = model
        self._data = data
        self._ns = namespace

        self.rotors = {
            namespace + local: self._lookup(mujoco.mjtObj.mjOBJ_ACTUATOR, local)
            for local in self._rotor_locals
        }

        self._body_id = self._lookup(mujoco.mjtObj.mjOBJ_BODY, self.BODY_NAME)
        joint_id = self._lookup(mujoco.mjtObj.mjOBJ_JOINT, self.FREEJOINT_NAME)
        self._qpos_adr = int(model.jnt_qposadr[joint_id])
        self._qvel_adr = int(model.jnt_dofadr[joint_id])
        self._imu_site_id = self._lookup(mujoco.mjtObj.mjOBJ_SITE, self.IMU_SITE_NAME)
        self.mount_site_id = self._lookup(mujoco.mjtObj.mjOBJ_SITE, self.MOUNT_SITE_NAME)

        self._sensors = {}
        for local in self._sensor_locals:
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, namespace + local)
            if sensor_id >= 0:
                self._sensors[local] = (int(model.sensor_adr[sensor_id]), int(model.sensor_dim[sensor_id]))

    def _lookup(self, obj_type, local_name: str) -> int:
        full_name = self._ns + local_name
        obj_id = mujoco.mj_name2id(self._model, obj_type, full_name)
        if obj_id < 0:
            raise RuntimeError(f"Multirotor: 编译产物中缺少必需的元素 '{full_name}'")
        return obj_id

    def _require_bound(self) -> None:
        if self._model is None or self._data is None:
            raise RuntimeError(
                "Multirotor 尚未绑定编译产物：请经 Environment.attach_robot() "
                "挂载编译后再调用"
            )

    # ---------- 属性 ----------

    @property
    def n_rotors(self) -> int:
        return len(self._rotor_locals)

    @property
    def rotor_names(self) -> list[str]:
        """旋翼全名列表（bind 后带命名空间前缀），按 rotor1..N 排序。"""
        return [self._ns + local for local in self._rotor_locals]

    @property
    def sensor_names(self) -> list[str]:
        """MJCF 中声明的传感器本地名列表。"""
        return list(self._sensor_locals)

    @property
    def model(self):
        """绑定的共享 MjModel（只读句柄，供控制器读取几何/质量/gear）。"""
        return self._model

    @property
    def data(self):
        """绑定的共享 MjData（只读句柄，供帧边界写入 ctrl）。"""
        return self._data

    @property
    def body_id(self) -> int:
        self._require_bound()
        return self._body_id

    # ---------- 控制写入 ----------

    def set_actuator(self, u) -> None:
        """按 rotor1..rotorN 顺序写入旋翼 actuator 推力 [N]。"""
        self._require_bound()
        values = np.asarray(u, dtype=float).reshape(-1)
        if values.size != self.n_rotors:
            raise ValueError(f"Multirotor.set_actuator: 期望 {self.n_rotors} 个推力值，收到 {values.size} 个")
        for local, value in zip(self._rotor_locals, values):
            self._data.ctrl[self.rotors[self._ns + local]] = float(value)

    # ---------- 状态读取 ----------

    def get_sensor(self, key: str) -> np.ndarray:
        """按约定本地名读取传感器数据（如 'gyro' / 'accel' / 'drone_pos'）。"""
        self._require_bound()
        if key not in self._sensors:
            raise KeyError(f"Multirotor: 传感器 '{self._ns}{key}' 不存在（已发现: {sorted(self._sensors)}）")
        adr, dim = self._sensors[key]
        return np.array(self._data.sensordata[adr: adr + dim], dtype=float)

    def get_state(self) -> SensorData:
        """从自由关节 qpos/qvel 提取平台位姿与速度。"""
        self._require_bound()
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

    def get_imu(self) -> dict[str, np.ndarray]:
        """读取 IMU 数据：陀螺仪、加速度计（以及磁强计，若模型提供）。"""
        imu = {
            "gyro": self.get_sensor("gyro"),
            "accel": self.get_sensor("accel"),
        }
        for optional_key in ("mag",):
            if optional_key in self._sensors:
                imu[optional_key] = self.get_sensor(optional_key)
        return imu

    def get_mount_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """机械臂挂载点的世界位姿（位置 + 3x3 旋转矩阵）。"""
        self._require_bound()
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
        self._require_bound()
        drone_pos = self._data.xpos[self._body_id]
        drone_mat = self._data.xmat[self._body_id].reshape(3, 3)
        positions = np.zeros((self.n_rotors, 3))
        yaw_coeffs = np.zeros(self.n_rotors)
        ctrl_ranges = np.zeros((self.n_rotors, 2))
        for i, local in enumerate(self._rotor_locals):
            act_id = self.rotors[self._ns + local]
            site_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_SITE, self._ns + local + "_site"
            )
            world_rel = self._data.site_xpos[site_id] - drone_pos
            positions[i] = drone_mat.T @ world_rel
            yaw_coeffs[i] = float(self._model.actuator_gear[act_id, 5])
            ctrl_ranges[i] = self._model.actuator_ctrlrange[act_id]
        return positions, yaw_coeffs, ctrl_ranges
