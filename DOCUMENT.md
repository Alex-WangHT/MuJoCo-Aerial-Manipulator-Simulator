# UAMSim 使用文档

> 本文档面向需要使用 UAMSim 框架进行多旋翼 / 空中机械臂仿真与控制的开发者，涵盖从快速上手到深入定制的全部内容。

---

## 目录

- [1. 架构概览](#1-架构概览)
  - [1.1 三线程实时模型](#11-三线程实时模型)
  - [1.2 两阶段组件生命周期](#12-两阶段组件生命周期)
  - [1.3 帧同步通信机制](#13-帧同步通信机制)
- [2. MJCF 模型命名约定](#2-mjcf-模型命名约定)
  - [2.1 场景环境（environment.xml）](#21-场景环境environmentxml)
  - [2.2 多旋翼平台（multirotor.xml）](#22-多旋翼平台multirotorxml)
  - [2.3 机械臂（manipulator.xml）](#23-机械臂manipulatorxml)
- [3. 启动仿真](#3-启动仿真)
  - [3.1 纯多旋翼仿真](#31-纯多旋翼仿真)
  - [3.2 多旋翼 + 机械臂仿真](#32-多旋翼--机械臂仿真)
  - [3.3 无头模式与加速](#33-无头模式与加速)
- [4. 编写控制器](#4-编写控制器)
  - [4.1 多旋翼控制器](#41-多旋翼控制器)
  - [4.2 机械臂控制器](#42-机械臂控制器)
  - [4.3 运行时更新控制目标](#43-运行时更新控制目标)
  - [4.4 混控器与旋翼几何](#44-混控器与旋翼几何)
- [5. 场景交互](#5-场景交互)
  - [5.1 障碍物查询](#51-障碍物查询)
  - [5.2 碰撞检测](#52-碰撞检测)
- [6. 遥测与感知](#6-遥测与感知)
  - [6.1 TelemetryBuffer（进程内缓存）](#61-telemetrybuffer进程内缓存)
  - [6.2 UDP 遥测通道](#62-udp-遥测通道)
  - [6.3 感知源（相机/雷达预留接口）](#63-感知源相机雷达预留接口)
- [7. API 速查](#7-api-速查)
  - [7.1 Environment](#71-environment)
  - [7.2 Multirotor](#72-multirotor)
  - [7.3 Manipulator](#73-manipulator)
  - [7.4 Robot](#74-robot)
  - [7.5 MujocoSimulation](#75-mujocosimulation)
  - [7.6 MultirotorController](#76-multirotorcontroller)
  - [7.7 ManipulatorController](#77-manipulatorcontroller)
- [8. 常见问题](#8-常见问题)

---

## 1. 架构概览

### 1.1 三线程实时模型

UAMSim 的核心设计理念是**仿真线程独占 `mjData`**，控制器线程绝不直接触碰 MuJoCo 数据。这样可以完全避免 Python GIL 与 MuJoCo 内部状态的数据竞争。

```
┌─────────────────────────────────────────────────────────────┐
│                      仿真线程 (MujocoSimulation)               │
│  ┌─────────────┐    发布快照    ┌─────────────────────────┐  │
│  │  mj_step()  │ ────────────> │ FrameMailbox.publish()  │  │
│  │  (1 kHz)    │               │ wait_done() 阻塞等回执   │  │
│  └─────────────┘               └─────────────────────────┘  │
│         ▲                                    │               │
│         │                                    │ 控制器写缓冲   │
│         │              flush()               ▼               │
│  ┌──────┴──────────────────────────────────────────┐        │
│  │  data.ctrl[] <- RotorActuator / ServoActuator    │        │
│  └──────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
  │ 多旋翼控制   │      │ 机械臂控制   │      │ 遥测/感知   │
  │ 器线程      │      │ 器线程      │      │ 后台线程    │
  └─────────────┘      └─────────────┘      └─────────────┘
```

| 线程 | 职责 | 对 `mjData` 的访问 |
|------|------|-------------------|
| **仿真线程** | 推进物理、发布快照、flush 执行器缓冲、渲染 | **唯一读写** |
| **多旋翼控制器线程** | 接收快照 → 计算推力 → 写入 `RotorActuator` 缓冲 | **只读快照，不碰 `mjData`** |
| **机械臂控制器线程** | 接收快照 → 计算关节角 → 写入 `ServoActuator` 缓冲 | **只读快照，不碰 `mjData`** |

### 1.2 两阶段组件生命周期

每个机器人组件（`Multirotor` / `Manipulator`）都有明确的两个阶段：

**阶段一：读取 MJCF（构造时）**

```python
multirotor = ua.Multirotor("path/to/multirotor.xml")
# 此时 multirotor.model 为 None，只完成了 MJCF 解析与命名自省
```

- 调用 `MjSpec.from_file()` 加载 XML
- 从 spec 中自省 actuator、sensor、body、joint、site 等约定信息
- 校验命名是否符合约定（如 `rotor1` .. `rotorN`）
- **此时不依赖任何 `MjModel` / `MjData`**

**阶段二：绑定编译产物（`attach_robot` 后）**

```python
env = ua.Environment("path/to/environment.xml")
uam = env.attach_robot(multirotor, manipulator)
# 此时 multirotor.model / data 已绑定，所有读写接口可用
```

- `Environment.attach_robot()` 把组件 spec 组合进场景
- 调用 `MjSpec.compile()` 统一编译
- 回调各组件的 `bind(model, data, namespace)` 解析 id 与地址
- 此后 `set_actuator()`、`get_sensor()`、`get_state()` 等接口可用

### 1.3 帧同步通信机制

仿真线程与控制器线程之间的通信由 `ControllerChannel` 封装，分两个方向：

**方向一：仿真 → 控制器（`FrameMailbox`）**

每物理帧执行一次：

1. 仿真线程 `publish(snapshot)` 发布当前帧传感快照 → 获得帧号
2. 仿真线程 `wait_done()` 阻塞，等待控制器回执
3. 控制器线程 `wait_snapshot()` 收到新帧快照
4. 控制器线程计算控制指令，写入执行器缓冲
5. 控制器线程 `mark_done(frame)` 回执
6. 仿真线程被唤醒，`flush()` 把缓冲统一落盘到 `data.ctrl`
7. 调用 `mj_step()` 推进物理

**方向二：控制器 → 仿真（执行器缓冲）**

控制器线程只修改**缓冲值**（纯 Python float），仿真线程在帧边界统一 `flush()` 写入 `data.ctrl`。`flush()` **只能由仿真线程调用**。

---

## 2. MJCF 模型命名约定

UAMSim 通过**名称自省**（name introspection）识别模型中的关键元素，因此 MJCF 文件必须遵循以下命名约定。

### 2.1 场景环境（environment.xml）

场景环境是"父模型"，负责提供地板、灯光、障碍物和机器人挂载点。

```xml
<mujoco model="flight_field">
  <!-- 关键约定：
    - 机器人挂载点 site:  uam_spawn（attach_robot 以此为放置位置）
    - 障碍物 geom:        名称以 obstacle_ 前缀开头（位置/接触查询）
    - 着陆台 geom:        landing_pad
    - 地板 geom:          floor
  -->
  <worldbody>
    <site name="uam_spawn" pos="0 0 0" size="0.05"/>
    <geom name="obstacle_pillar_a" type="cylinder" pos="1.5 0.5 0.5" size="0.12 0.5"/>
    <geom name="floor" type="plane" size="10 10 0.02"/>
  </worldbody>
</mujoco>
```

| 元素 | 名称 | 说明 |
|------|------|------|
| 机器人挂载点 site | `uam_spawn` | 机器人整机 attach 到该 site |
| 障碍物 geom | `obstacle_*` | 名称以 `obstacle_` 前缀开头 |

### 2.2 多旋翼平台（multirotor.xml）

多旋翼平台是"子模型"，通过 `attach` 挂到场景的 `uam_spawn` 上。

```xml
<mujoco model="multirotor_platform">
  <worldbody>
    <body name="drone" pos="0 0 1.5">
      <freejoint name="drone_free"/>
      <site name="imu_site" pos="0 0 0" size="0.008"/>
      <site name="manipulator_mount" pos="0 0 -0.05" size="0.01"/>
      <!-- 旋翼 body + site ... -->
      <body name="rotor1_body" pos="0.34 0 0">
        <site name="rotor1_site" pos="0 0 0" size="0.01"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="rotor1" site="rotor1_site" gear="0 0 1 0 0 0.02" ctrlrange="0 12"/>
    <motor name="rotor2" site="rotor2_site" gear="0 0 1 0 0 -0.02" ctrlrange="0 12"/>
    <!-- ... rotorN -->
  </actuator>

  <sensor>
    <gyro name="gyro" site="imu_site"/>
    <accelerometer name="accel" site="imu_site"/>
    <framepos name="drone_pos" objtype="body" objname="drone"/>
    <framequat name="drone_quat" objtype="body" objname="drone"/>
  </sensor>
</mujoco>
```

| 类别 | 命名约定 | 说明 |
|------|----------|------|
| 机体 body | `drone` | 必须有 |
| 自由关节 | `drone_free` | 必须有，用于平台自由飞行 |
| IMU site | `imu_site` | 陀螺仪/加速度计/磁力计挂载点 |
| 机械臂挂载点 | `manipulator_mount` | 机械臂 attach 到该 site |
| 旋翼 actuator | `rotor1` .. `rotorN` | 按编号顺序，site 传动，gear 含 Z 轴反扭矩 |
| 旋翼 site | `rotorN_site` | 每个旋翼对应一个 site |
| 传感器（可选） | `gyro` / `accel` / `mag` / `drone_pos` / `drone_quat` / `mount_pos` / `mount_quat` | 缺什么跳过什么 |

**旋翼 gear 说明**：`gear="0 0 1 0 0 tz"`，其中 `tz` 是 Z 轴反扭矩系数。相邻旋翼应符号交替（模拟正/反转桨）。

### 2.3 机械臂（manipulator.xml）

机械臂也是"子模型"，通过 `attach` 挂到多旋翼的 `manipulator_mount` 上。

```xml
<mujoco model="manipulator_3dof">
  <worldbody>
    <body name="Base">
      <joint name="joint1" axis="0 1 0" range="-1.92 1.92"/>
      <body name="Upper_Arm">
        <joint name="joint2" axis="1 0 0" range="-3.32 0.174"/>
        <body name="Lower_Arm">
          <joint name="joint3" axis="1 0 0" range="-0.174 3.14"/>
          <site name="ee_site" pos="0 0.0052 0.1349" size="0.008"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="joint1" joint="joint1" kp="50"/>
    <position name="joint2" joint="joint2" kp="50"/>
    <position name="joint3" joint="joint3" kp="50"/>
  </actuator>

  <sensor>
    <jointpos name="jointpos1" joint="joint1"/>
    <jointvel name="jointvel1" joint="joint1"/>
    <framepos name="ee_pos" objtype="site" objname="ee_site"/>
    <framequat name="ee_quat" objtype="site" objname="ee_site"/>
  </sensor>
</mujoco>
```

| 类别 | 命名约定 | 说明 |
|------|----------|------|
| 关节 | `joint1` .. `jointN` | 非 free 关节，按运动树顺序 |
| 舵机 | 与关节同名 | `<position joint="jointN">`，bind 时通过 `actuator_trnid` 反查 |
| 末端 site | `ee_site` | 末端执行器位置 |
| 传感器（可选） | `jointposN` / `jointvelN` / `ee_pos` / `ee_quat` | 缺什么跳过什么 |

---

## 3. 启动仿真

### 3.1 纯多旋翼仿真

```python
import threading
import numpy as np
import UAMSim as ua

# 1. 构造场景与组件
shutdown = threading.Event()
env = ua.Environment("examples/models/environment.xml")
multirotor = ua.Multirotor("examples/models/multirotor.xml")

# 2. 启动仿真线程
sim = ua.MujocoSimulation(
    shutdown, env, multirotor, None,          # None = 无机械臂
    use_viewer=True,                          # 开启 MuJoCo viewer
    real_time_factor=1.0,                     # 实时
)
sim.start()
assert sim.wait_ready(timeout=10.0), "场景编译超时"

# 3. 构造控制器
class Hover(ua.MultirotorController):
    def controller(self, feedback, target_z, hover):
        err = target_z - feedback.dronePosition[2]
        return np.full(self._multirotor.n_rotors, hover + 4.0 * err)

ctrl = Hover(
    sim.uam.multirotor, sim.drone_channel, shutdown,
    target_z=1.5, hover=sim.uam.hover_thrust(),
)
ctrl.start()

# 4. 放行物理
sim.start_physics()
```

### 3.2 多旋翼 + 机械臂仿真

```python
# 构造场景与组件
shutdown = threading.Event()
env = ua.Environment("examples/models/environment.xml")
multirotor = ua.Multirotor("examples/models/multirotor.xml")
manipulator = ua.Manipulator("examples/models/Manipulator.xml")

# 启动仿真（传入 manipulator）
sim = ua.MujocoSimulation(
    shutdown, env, multirotor, manipulator,
    use_viewer=True,
)
sim.start()
sim.wait_ready()

# 多旋翼控制器
class DroneCtrl(ua.MultirotorController):
    def controller(self, feedback, hover):
        return np.full(self._multirotor.n_rotors, hover)

drone_ctrl = DroneCtrl(sim.uam.multirotor, sim.drone_channel, shutdown,
                       hover=sim.uam.hover_thrust())

# 机械臂控制器
class ArmCtrl(ua.ManipulatorController):
    def controller(self, feedback, q_target):
        return q_target

arm_ctrl = ArmCtrl(sim.uam.manipulator, sim.arm_channel, shutdown,
                   q_target=np.zeros(sim.uam.manipulator.n_joints))

drone_ctrl.start()
arm_ctrl.start()
sim.start_physics()
```

### 3.3 无头模式与加速

```python
# 不开 viewer，5× 加速，适合批量测试
sim = ua.MujocoSimulation(
    shutdown, env, multirotor, manipulator,
    use_viewer=False,           # 无头模式
    real_time_factor=5.0,       # 5× 加速（物理帧每 0.2 ms 执行一次）
)
```

> 注意：`real_time_factor` 只影响**墙钟对齐**，不影响物理精度。MuJoCo 的物理步长由 XML 中的 `<option timestep="0.001"/>` 决定。

---

## 4. 编写控制器

### 4.1 多旋翼控制器

继承 `MultirotorController`，唯一需要重写的是 `controller()` 方法。

```python
class MyDroneCtrl(ua.MultirotorController):
    def controller(self, feedback, target_pos, kp, kd, hover):
        """
        参数:
            feedback: SensorSnapshot，包含以下字段:
                - timestamp         # 当前仿真时间 [s]
                - timestep          # 物理步长 [s]
                - dronePosition     # 世界坐标位置 [m] (3,)
                - droneVelocity     # 世界坐标速度 [m/s] (3,)
                - droneOrientation  # roll-pitch-yaw 欧拉角 [rad] (3,)
                - droneAngularVelocity  # 机体角速度 [rad/s] (3,)
            target_pos: 目标位置（构造时传入的 kwargs）
            kp, kd: 增益参数（构造时传入的 kwargs）
            hover: 悬停推力（构造时传入的 kwargs）

        返回:
            (n_rotors,) ndarray，各旋翼推力 [N]
        """
        err = target_pos - feedback.dronePosition
        vel = feedback.droneVelocity
        thrust = hover + kp * err[2] - kd * vel[2]
        return np.full(self._multirotor.n_rotors, thrust)
```

**构造与启动**：

```python
ctrl = MyDroneCtrl(
    sim.uam.multirotor,       # 已绑定的 Multirotor 组件
    sim.drone_channel,        # 仿真线程暴露的通道
    shutdown,                 # 共享退出事件
    target_pos=np.array([0.0, 0.0, 1.5]),
    kp=4.0, kd=3.0,
    hover=sim.uam.hover_thrust(),
)
ctrl.start()
```

**继承约束**：框架在语法层面封死了 `__init__` 和 `run` 的重写，子类**只能**重写 `controller()`。自定义状态（参考值、增益、滤波器等）一律通过构造 `kwargs` 传入，由 `controller()` 的自定义参数接收。

### 4.2 机械臂控制器

与多旋翼控制器完全同构，只是反馈字段和返回值不同。

```python
class MyArmCtrl(ua.ManipulatorController):
    def controller(self, feedback, q_target, kp):
        """
        参数:
            feedback: SensorSnapshot，包含以下字段:
                - timestamp         # 当前仿真时间 [s]
                - timestep          # 物理步长 [s]
                - jointPositions    # 关节角 [rad] (n_joints,)
                - jointVelocities   # 关节角速度 [rad/s] (n_joints,)
                - eePosition        # 末端执行器世界位置 [m] (3,)
                - eeJacobian        # 末端位置雅可比 (3, n_joints)
            q_target: 目标关节角（构造时传入）
            kp: 位置增益（构造时传入）

        返回:
            (n_joints,) ndarray，各关节目标角 [rad]
        """
        err = q_target - feedback.jointPositions
        return q_target + kp * err
```

**构造与启动**：

```python
ctrl = MyArmCtrl(
    sim.uam.manipulator,      # 已绑定的 Manipulator 组件
    sim.arm_channel,          # 仿真线程暴露的通道（纯多旋翼时 None）
    shutdown,
    q_target=np.array([0.0, -0.5, 0.5]),
    kp=1.0,
)
ctrl.start()
```

### 4.3 运行时更新控制目标

控制器支持**线程安全**的目标参数热更新：

```python
# 构造时设定初始目标
ctrl = HoverController(..., target_pos=np.array([0, 0, 1.5]), kp=4.0)

# 仿真运行期间随时更新（下一帧生效）
ctrl.set_target(target_pos=np.array([0, 0, 2.0]))
ctrl.set_target(kp=5.0, kd=4.0)

# 读取当前生效参数
print(ctrl.get_target())  # {'target_pos': array([0, 0, 2.0]), 'kp': 5.0, 'kd': 4.0, ...}
```

`set_target()` 传入的参数优先级高于构造时的参数，但不会修改原始 `self.params`。子类的 `controller()` 签名无需改动。

### 4.4 混控器与旋翼几何

多旋翼控制律通常输出 `[总推力, τx, τy, τz]`，需要混控矩阵映射到各旋翼推力。

```python
# 从旋翼几何自动构造混控伪逆矩阵
positions, yaw_coeffs, ctrl_ranges = multirotor.get_rotor_geometry()
# positions:   (N, 3)  各旋翼在机体系下的坐标
# yaw_coeffs:  (N,)    各旋翼单位推力的 Z 反扭矩系数
# ctrl_ranges: (N, 2)  各旋翼推力上下限 [N]

# 构造分配矩阵并求伪逆
alloc = np.vstack([
    np.ones(multirotor.n_rotors),
    positions[:, 1],      # y 坐标 -> roll 力矩
    -positions[:, 0],     # -x 坐标 -> pitch 力矩
    yaw_coeffs,           # Z 反扭矩 -> yaw 力矩
])
mixer = alloc.T @ np.linalg.inv(alloc @ alloc.T)   # (N, 4)

# 使用：u = mixer @ [thrust, tau_x, tau_y, tau_z]
```

---

## 5. 场景交互

### 5.1 障碍物查询

```python
# 获取所有障碍物名称
print(env.obstacles)
# {'obstacle_pillar_a': 10, 'obstacle_box': 12, ...}

# 获取障碍物世界坐标
positions = env.get_obstacle_positions()
# {'obstacle_pillar_a': array([1.5, 0.5, 0.5]), ...}
```

### 5.2 碰撞检测

```python
# 获取全部接触对
contacts = env.get_contacts()
# [(geom1_name, geom2_name, contact_point), ...]

# 只获取涉及机器人的接触
robot_contacts = env.get_robot_contacts(prefix="uam/")
```

---

## 6. 遥测与感知

### 6.1 TelemetryBuffer（进程内缓存）

在仿真线程内收集数据，供主线程或监控逻辑查询：

```python
telemetry = ua.TelemetryBuffer(max_samples=5000)

sim = ua.MujocoSimulation(
    shutdown, env, multirotor, None,
    telemetry_buffer=telemetry,
)
sim.start()

# ... 仿真运行中 ...

# 主线程随时查询
t_arr, pos_arr, eul_arr = telemetry.snapshot()
# t_arr:    (N,)   时间戳
# pos_arr:  (N, 3) 位置
# eul_arr:  (N, 3) 欧拉角
```

### 6.2 UDP 遥测通道

把仿真状态实时发往另一个进程（如可视化面板、日志记录器）：

```python
sim = ua.MujocoSimulation(
    shutdown, env, multirotor, manipulator,
    telemetry_target=("127.0.0.1", 9100),   # UDP 目标地址
    telemetry_rate_hz=100.0,                # 发布频率
)
```

**接收端示例**（任意语言）：

```python
import socket, json

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 9100))
while True:
    data, _ = sock.recvfrom(2048)
    packet = json.loads(data.decode())
    # packet = {"t": 1.234, "pos": [x, y, z], "rpy": [r, p, y], ...}
```

协议：每个 UDP 数据报 = 一行 JSON（UTF-8，`\n` 结尾），字段包括：

| 字段 | 说明 |
|------|------|
| `t` | 仿真时间 [s] |
| `pos` | 位置 [m] `[x, y, z]` |
| `vel` | 速度 [m/s] `[vx, vy, vz]` |
| `rpy` | 欧拉角 [rad] `[roll, pitch, yaw]` |
| `omega` | 角速度 [rad/s] `[ωx, ωy, ωz]` |
| `thrusts` | 各旋翼推力 [N] |
| `joints` | 关节角 [rad]（有机械臂时） |
| `ee` | 末端位置 [m]（有机械臂时） |

### 6.3 感知源（相机/雷达预留接口）

框架预留了感知源接口，可接入相机、雷达等大数据量传感器：

```python
class MyCamera(ua.PerceptionSource):
    name = "front_camera"
    rate_hz = 30.0

    def capture(self, model, data):
        # 在仿真线程内执行，可访问 mjData
        # 返回 (payload_bytes, meta_dict)
        image = render_camera(model, data)   # 伪代码
        return image.tobytes(), {"shape": [480, 640, 3], "dtype": "uint8"}

camera = MyCamera()
sim.add_perception_source(camera)

# 启动感知 UDP 通道
sim = ua.MujocoSimulation(
    ...,
    perception_target=("127.0.0.1", 9200),
)
```

感知数据通过 UDP 分片发送（允许丢帧，绝不阻塞物理主循环）。接收端用 `PerceptionReceiver` 重组：

```python
receiver = ua.PerceptionReceiver(port=9200)
receiver.start()
meta, payload = receiver.recv(timeout=1.0)
```

---

## 7. API 速查

### 7.1 Environment

```python
env = ua.Environment("path/to/environment.xml")

# 组合与编译
robot = env.attach_robot(multirotor, manipulator=None,
                         site="uam_spawn", prefix="uam/",
                         mount_site="manipulator_mount", arm_prefix="arm/")

# 仿真推进
env.step(n_steps=1)
env.reset()

# 场景查询
env.obstacles                              # dict: 障碍物名称 -> geom_id
env.get_obstacle_positions()               # dict: 障碍物名称 -> 世界位置
env.get_contacts()                         # list: (geom1, geom2, contact_point)
env.get_robot_contacts(prefix="uam/")      # 只保留涉及机器人的接触

# 编译后可用
env.model                                  # MjModel
env.data                                   # MjData
```

### 7.2 Multirotor

```python
m = ua.Multirotor("path/to/multirotor.xml")

# 属性（bind 后）
m.n_rotors                                 # 旋翼数量
m.rotor_names                              # 旋翼全名列表（含前缀）
m.sensor_names                             # 传感器本地名列表
m.model, m.data                            # 共享 MjModel / MjData
m.body_id                                  # 机体 body id

# 控制写入
m.set_actuator(u)                          # (n_rotors,) 推力 [N]

# 状态读取
m.get_sensor("gyro")                       # 陀螺仪读数
m.get_sensor("accel")                      # 加速度计读数
m.get_sensor("drone_pos")                  # 机体位置
m.get_sensor("drone_quat")                 # 机体四元数
m.get_state()                              # SensorSnapshot（完整状态）
m.get_imu()                                # dict: {"gyro": ..., "accel": ...}
m.get_mount_pose()                         # (pos, rot_mat) 挂载点位姿
m.get_rotor_geometry()                     # (positions, yaw_coeffs, ctrl_ranges)
```

### 7.3 Manipulator

```python
arm = ua.Manipulator("path/to/manipulator.xml")

# 属性（bind 后）
arm.n_joints                               # 关节数量
arm.joint_names                            # 关节全名列表（含前缀，按运动树序）
arm.sensor_names                           # 传感器本地名列表
arm.ee_site_id                             # 末端 site id

# 控制写入
arm.set_actuator(q)                        # (n_joints,) 关节目标角 [rad]

# 状态读取
arm.get_joint_positions()                  # (n_joints,) 当前关节角
arm.get_joint_velocities()                 # (n_joints,) 当前关节角速度
arm.get_ee_pose()                          # (pos, rot_mat) 末端位姿
arm.get_sensor("jointpos1")                # 指定传感器读数
arm.get_state()                            # SensorSnapshot（含雅可比）
```

### 7.4 Robot

```python
uam = env.attach_robot(multirotor, manipulator)

uam.multirotor                             # Multirotor 组件视图
uam.manipulator                            # Manipulator 组件视图（纯多旋翼时为 None）
uam.has_manipulator                        # bool
uam.prefix                                 # 名称前缀（如 "uam/"）
uam.model, uam.data                        # 共享 MjModel / MjData
uam.hover_thrust()                         # 悬停单旋翼推力 [N]
```

### 7.5 MujocoSimulation

```python
sim = ua.MujocoSimulation(
    shutdown_event,                          # threading.Event
    environment,                             # Environment 实例
    multirotor,                              # Multirotor 实例
    manipulator=None,                        # Manipulator 实例或 None
    telemetry_buffer=None,                   # TelemetryBuffer 或 None
    fixed_rotor_thrust=None,                 # 固定开环推力（调试）
    use_viewer=True,                         # 是否开 viewer
    real_time_factor=1.0,                    # 实时倍率
    telemetry_target=None,                   # (host, port) UDP 遥测目标
    telemetry_rate_hz=100.0,                 # 遥测频率
    perception_target=None,                  # (host, port) UDP 感知目标
)

sim.start()                                  # 启动仿真线程
sim.wait_ready(timeout=None)                 # 等待编译完成，返回 bool
sim.start_physics()                          # 放行物理推进
sim.stop()                                   # 设置退出标志
sim.join(timeout=None)                       # 等待线程结束

# 编译后暴露
sim.uam                                      # Robot 句柄
sim.drone_channel                            # ControllerChannel（多旋翼）
sim.arm_channel                              # ControllerChannel（机械臂，可能为 None）

# 感知源注册（物理启动前调用）
sim.add_perception_source(source)            # source: PerceptionSource 实例
```

### 7.6 MultirotorController

```python
class MyCtrl(ua.MultirotorController):
    def controller(self, feedback, **params):
        # feedback: SensorSnapshot
        # params: 构造时传入的 kwargs
        return np.ndarray  # (n_rotors,) 推力 [N]

ctrl = MyCtrl(
    multirotor,                              # 已绑定的 Multirotor
    channel,                                 # sim.drone_channel
    shutdown_event=None,                     # threading.Event
    **controller_params,                     # 自定义参数，逐帧透传
)

ctrl.start()                                 # 启动控制器线程
ctrl.set_target(**kwargs)                    # 线程安全更新目标
ctrl.get_target()                            # 读取当前生效参数
```

### 7.7 ManipulatorController

```python
class MyCtrl(ua.ManipulatorController):
    def controller(self, feedback, **params):
        # feedback: SensorSnapshot
        return np.ndarray  # (n_joints,) 关节目标角 [rad]

ctrl = MyCtrl(
    manipulator,                             # 已绑定的 Manipulator
    channel,                                 # sim.arm_channel
    shutdown_event=None,
    **controller_params,
)

# 方法与 MultirotorController 相同
```

---

## 8. 常见问题

### Q: 控制器 `controller()` 抛异常会怎样？

A: 控制环会**保持上一条缓冲指令**，照常回执，避免仿真线程死等。错误内容变化时打印一次警告。这是设计行为，保证单点故障不会击落整个仿真。

### Q: 组件已经绑定过，无法重复 attach？

A: `Multirotor` / `Manipulator` 一经 `attach_robot()` 编译即绑定到场景，不可复用。如需重新挂载，请**重新构造实例**：

```python
# 错误：multirotor 已经绑定
env1.attach_robot(multirotor)
env2.attach_robot(multirotor)  # ValueError!

# 正确：重新构造
env1.attach_robot(ua.Multirotor(path))
env2.attach_robot(ua.Multirotor(path))
```

### Q: 控制器线程在 `sim.start_physics()` 之前还是之后启动？

A: **之前**。正确顺序是：

```python
sim.start(); sim.wait_ready()     # 1. 仿真线程启动，编译完成
ctrl.start()                       # 2. 控制器线程启动（注册到通道）
sim.start_physics()                # 3. 放行物理
```

如果控制器在 `start_physics()` 之后启动，会错过前几帧的同步窗口，但通道机制会自动从下一帧开始介入。

### Q: 如何关闭 viewer？

A: 直接关闭 viewer 窗口即可，仿真线程检测到 `viewer.is_running() == False` 后会优雅退出，并关闭通道唤醒控制器线程。

### Q: 纯多旋翼模式下机械臂相关 API 会怎样？

A: `MujocoSimulation` 的 `arm_channel` 为 `None`，`sim.uam.manipulator` 为 `None`，`sim.uam.has_manipulator` 为 `False`。试图访问 `manipulator` 的接口会抛出 `AttributeError`。

### Q: `real_time_factor` 为 5.0 时物理精度会下降吗？

A: **不会**。`real_time_factor` 只影响**墙钟对齐**（每帧等待多久），不影响 MuJoCo 的物理步长。物理精度完全由 XML 中的 `timestep` 决定。设置为 `5.0` 意味着仿真以 5 倍实时速度运行（每帧等待时间缩短为 1/5）。

### Q: 可以同时挂载多个机器人吗？

A: `Environment.attach_robot()` 设计支持多机器人，每个机器人有独立的 `prefix`（如 `"uam1/"`、`"uam2/"`）。但当前 `MujocoSimulation` 只接收单个 multirotor + manipulator 对。多机器人场景需要自定义仿真循环。
