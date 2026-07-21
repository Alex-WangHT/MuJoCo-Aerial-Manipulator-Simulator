# Architecture

> 本文档以当前 `src/` 实际代码为准，描述分层结构、模型组合机制、
> 三线程运行时模型与调用关系。

## 1. 目录结构

```text
.
├── main.py                        # 程序入口（三线程接线）
├── requirements.txt               # numpy / mujoco>=3.2 / glfw / matplotlib
├── docs/
│   └── architecture.md            # 本文档
├── models/
│   ├── environment.xml            # 场景：地板/灯光/障碍物/着陆台/uam_spawn
│   ├── multirotor.xml             # 六旋翼平台（纯机器人本体，无场景元素）
│   └── Manipulator.xml            # 三自由度机械臂（子模型）
├── src/
│   ├── __init__.py                # 包导出 + quaternionToEuler + _MODELS_DIR
│   ├── Messages.py                # SensorData / ManipulatorSensorData / ControlInput
│   ├── Telemetry.py               # TelemetryBuffer（线程安全遥测环形缓冲）
│   ├── FrameSync.py               # FrameMailbox 帧同步邮箱 + ControllerChannel 通道
│   ├── Actuators.py               # RotorActuator / ServoActuator 执行器缓冲
│   ├── Multirotor.py              # 多旋翼组件（自读 MJCF，set_actuator/get_sensor）
│   ├── Manipulator.py             # 机械臂组件（自读 MJCF，同构）
│   ├── AerialManipulator.py       # 机器人组合器（输入为两组件实例）
│   ├── Environment.py             # 场景组合器（最顶层）
│   ├── MultirotorController.py    # 多旋翼控制器线程基类（默认串级 PID + 混控）
│   ├── ManipulatorController.py   # 机械臂控制器线程基类（默认关节/末端 DLS）
│   └── MujocoSimulation.py        # 仿真线程（仅对 Environment 编译，独占 mjData）
└── tests/                         # 四个端到端测试（见第 7 节）
```

包外代码一律从包根导入：`from src import MujocoSimulation`（`__init__.py`
再导出全部类与数据载体）；包内各文件之间用相对导入
（`from .Environment import Environment`）。

## 2. 三线程模型与调用关系

**Simulation 不接收 Controller 对象。** 三个线程通过帧同步通道解耦：

```text
main.py（主线程，接线）
  ├── threading.Event              全局退出信号
  ├── TelemetryBuffer              遥测缓冲（仿真线程写，GCS 读）
  ├── MujocoSimulation (thread)    唯一访问 mjData 的线程；输入 Environment +
  │     │                          AerialManipulator 实例，合并后仅对 Environment 编译
  │     ├── Environment            加载 environment.xml，最顶层组合器，统一编译
  │     │     └── AerialManipulator (compileModel=False)
  │     │           ├── Multirotor   读 multirotor.xml（bind namespace="uam/"）
  │     │           └── Manipulator  读 Manipulator.xml（bind namespace="uam/arm/"）
  │     ├── drone_channel          ControllerChannel（编译后暴露）
  │     └── arm_channel            ControllerChannel（纯多旋翼构型为 None）
  ├── MultirotorController (thread)  ──drone_channel──> 快照邮箱 + 旋翼执行器缓冲
  └── ManipulatorController (thread) ──arm_channel──> 快照邮箱 + 舵机执行器缓冲
```

接线顺序（`wait_ready` 到 `start_physics` 之间物理停放，控制器构造读
视图初始状态无数据竞争）：

```python
env = Environment()
uam = AerialManipulator(Multirotor(), Manipulator(), compileModel=False)
sim = MujocoSimulation(shutdown, env, uam, useViewer=False)   # 合并打包
sim.start()
sim.wait_ready()                       # 场景编译完成，通道已暴露
drone = MultirotorController(sim.drone_channel, sim.uam.multirotor, targetPosition=...)
arm = ManipulatorController(sim.arm_channel, sim.uam.manipulator, ...)
drone.start(); arm.start()             # 两个控制器各自独立线程
sim.start_physics()                    # 放行物理推进
```

**线程安全约定**：全进程只有仿真线程访问 mjData。控制器线程只读
帧快照（当帧拷贝）、只写执行器缓冲（纯 Python float）；仿真线程在
帧边界统一 `flush()` 进 `data.ctrl`。

**依赖方向单向向下**：组件只依赖包内 `Messages`；组合器依赖组件；
控制器基类依赖组件视图（几何/质量自省）与通道；仿真线程依赖组合器
与通道；入口依赖一切。

## 3. 模型组合机制（mjSpec）

所有组合发生在 **spec 层**，最终只编译**一个** MjModel —— 机械臂与平台、
机器人与场景共享同一套动力学，反作用力/接触天然耦合：

```text
environment.xml (父)
  └── Environment.attach_robot: attach(uam_spec, site="uam_spawn", prefix="uam/")
        multirotor.xml (Multirotor 读取)
          └── AerialManipulator: attach(arm_spec, site="manipulator_mount", prefix="arm/")
                Manipulator.xml (Manipulator 读取)
```

- 两级名称前缀：`uam/rotor1`、`uam/drone`、`uam/arm/joint1`、`uam/arm/ee_site`
- **组件两阶段生命周期**：`Multirotor` / `Manipulator` 构造时各自读取自身
  MJCF（`MjSpec.from_file`），立即从 spec 自省 actuator / sensor / 关节 /
  site 等约定信息并校验命名约定；统一编译后由组合器回调 `bind(model, data,
  namespace)`，把名称解析为共享模型中的 id 与地址，此后 `set_actuator` /
  `get_sensor` 等读写接口可用
- `AerialManipulator` 两种模式：独立运行（`compileModel=True`，自身编译、
  无前缀）或放入场景（`compileModel=False`，由 Environment 编译后经
  `bind_views(model, data, namespace)` 绑定组件）
- 挂载关系全部声明在 MJCF：机械臂固连位置 = multirotor.xml 中
  `manipulator_mount` site（机体系 `pos="0 0 -0.05"`）；机器人出生位置 =
  environment.xml 中 `uam_spawn` site——Python 中不硬编码坐标
- 场景元素（地板/灯光/障碍物）只属于 environment.xml；机器人 MJCF 保持纯净
- 模型目录常量 `_MODELS_DIR` 定义在包 `__init__.py`（指向仓库根 `models/`）

## 4. 模块职责与公开 API

以下类均位于 `src/` 包内，并从包根再导出。

### 数据载体（`Messages.py` / `Telemetry.py`）

- `SensorData`：timestamp/timestep/位置/速度/欧拉角/角速度（多旋翼快照）
- `ManipulatorSensorData`：timestamp/timestep/关节角/关节角速度/末端位置/
  末端雅可比切片 `eeJacobian` (3, n_joints)（机械臂快照；雅可比依赖 mjData，
  由仿真线程随帧算好发布）
- `ControlInput`：控制器输出，旋翼推力向量 `u` [N]
- `TelemetryBuffer(maxSamples=5000)`：锁保护环形缓冲，`append/snapshot/clear`，
  跨线程（仿真写 → GCS 读）安全

### 线程通信（`FrameSync.py` / `Actuators.py`）

- `FrameMailbox`：帧同步邮箱（单发布者/单订阅者）。仿真线程
  `publish(snapshot)` → `wait_done(shutdown)` 阻塞到回执；控制器线程
  `wait_snapshot(last_frame)` 阻塞到新帧 → 处理 → `mark_done(frame)`；
  `close()` 后订阅者收到 `None` 并退出
- `ControllerChannel`：邮箱 + 执行器缓冲注册表。控制器构造时
  `attach(actuators)` 注册（即订阅）；仿真线程 `flush()` 在帧边界把全部
  缓冲写入 mjData；`close()` 随仿真退出关闭
- `RotorActuator` / `ServoActuator`：单通道执行器缓冲。
  控制器线程 `set_thrust(N)` / `set_target(rad)` 只写缓冲值（带范围截断）；
  `flush()` 只能由仿真线程调用

### `Environment.py` — class `Environment`

场景组合器与场景交互查询。

| 方法 | 说明 |
|---|---|
| `__init__(environmentPath=None)` | 加载场景 MJCF（默认 models/environment.xml） |
| `attach_robot(uam, site="uam_spawn", prefix="uam/")` | 挂载未编译机器人并统一编译 |
| `compile()` | 编译场景共享模型，绑定全部机器人组件，刷新障碍物表 |
| `step(n_steps=1)` / `reset()` | 推进 / 重置（mj_resetData + mj_forward） |
| `get_obstacle_positions()` | `obstacle_` 前缀 geom 的世界坐标 |
| `get_contacts()` | 全部接触对 `(geom1, geom2, pos)` |
| `get_robot_contacts(prefix="uam/")` | 只保留涉及指定机器人的接触对 |

属性：`model`、`data`、`robots`、`obstacles`。

### `AerialManipulator.py` — class `AerialManipulator`

机器人（UAM）组合器，**输入为组件实例**。

| 方法 | 说明 |
|---|---|
| `__init__(multirotor, manipulator=None, mountSite="manipulator_mount", armPrefix="arm/", compileModel=True)` | 接收已读 MJCF 的 `Multirotor` / `Manipulator` 实例，attach 打包为整机 spec；`manipulator=None` 即纯多旋翼构型 |
| `compile()` | 独立模式：编译自身 spec 并绑定无前缀组件 |
| `bind_views(model, data, namespace="")` | 绑定组件到（场景共享的）模型 |
| `step(n_steps=1)` / `reset()` | 推进 / 重置（共享模型时 reset 会重置整个场景） |
| `hover_thrust()` | 整机悬停单旋翼推力：总质量 × 9.81 / 旋翼数 |

属性：`spec`、`model`、`data`、`multirotor`、`manipulator`、`has_manipulator`。

### `Multirotor.py` — class `Multirotor`

多旋翼平台组件（**自己读取 MJCF**，两阶段生命周期）。
类常量：`BODY_NAME="drone"`、`FREEJOINT_NAME="drone_free"`、
`IMU_SITE_NAME="imu_site"`、`MOUNT_SITE_NAME="manipulator_mount"`。

- `__init__(multirotorPath=None)`：`MjSpec.from_file` 读取 multirotor.xml，
  从 spec 自省 actuator（`rotorN` 正则，按编号排序）与 sensor 清单，
  校验 body/joint/site 命名约定——此阶段不编译、不依赖 MjModel/MjData
- `bind(model, data, namespace="")`：统一编译后由组合器回调，把名称解析为
  共享模型中的 actuator id、`sensor_adr` + `sensor_dim`、自由关节地址等
- `set_actuator(u)`：按 rotor1..N 顺序写旋翼推力 [N]
- `get_sensor(key)` / `get_imu()`：gyro/accel（+mag）读数
- `get_state() -> SensorData`：自由关节 qpos/qvel → 位置/速度/欧拉角/角速度
- `get_mount_pose()`：挂载点世界位姿（位置 + 3×3 旋转矩阵）
- `get_rotor_geometry()`：旋翼机体系坐标 + 单位推力 Z 反扭矩系数
  （取自 gear）+ 推力范围——控制分配/混控所需几何
- 属性：`spec`、`n_rotors`、`rotor_names`、`sensor_names`、`body_id`、`model`、`data`

### `Manipulator.py` — class `Manipulator`

机械臂组件（**自己读取 MJCF**，两阶段生命周期，与 Multirotor 同构）。
类常量：`EE_SITE_NAME="ee_site"`。

- `__init__(manipulatorPath=None)`：`MjSpec.from_file` 读取 Manipulator.xml，
  从 spec 自省非 free 关节、舵机 actuator、sensor 清单，校验 `ee_site`
- `bind(model, data, namespace="arm/")`：统一编译后由组合器回调；舵机经
  `actuator_trnid` 反查绑定关节（`servos`，不依赖命名）
- `set_actuator(q)`：写各关节位置舵机目标角 [rad]
- `get_sensor(local_key)`：如 `'jointpos1'`、`'ee_pos'`
- `get_state() -> ManipulatorSensorData`：关节角/角速度 + 末端位置 +
  末端雅可比切片（供帧同步发布）
- `get_joint_positions()` / `get_joint_velocities()` / `get_ee_pose()`
- 属性：`spec`、`n_joints`、`joint_names`、`sensor_names`、`ee_site_id`、`model`、`data`

### `MultirotorController.py` — class `MultirotorController(threading.Thread)`

多旋翼控制器**线程基类**——用户自定义控制器继承本类，只重写
`control()` 一个方法即可；装配、协议、混控与推力截断全部由基类完成。

- `__init__(channel, multirotor, targetPosition=None, ...gains)`：
  构造即装配（物理停放期调用，读视图安全）——自省旋翼几何/质量/推力
  范围，构建 4×N 分配矩阵伪逆（`_alloc_pinv`、`_mass`、`_gravity`、
  `_ctrl_min/_ctrl_max` 供子类复用）；自动实例化 `RotorActuator` 列表
  （`self.actuators`，初始指令 = 均分悬停推力）并 `channel.attach` 注册
- `run()`（冻结）：`wait_snapshot -> control() -> mark_done` 循环，
  控制频率与物理帧严格同步；异常不杀线程，沿用上一帧缓冲指令
- `set_target_position(pos, yaw=None)`：目标注入点（供 GCS/MAVLink）
- 默认控制律 `control()`：串级 PID——位置环 PID → 期望加速度 →
  小角近似分配 roll/pitch → 姿态环 PID → 力矩 → 伪逆混控，
  推力写入 `self.actuators[i].set_thrust()`

### `ManipulatorController.py` — class `ManipulatorController(threading.Thread)`

机械臂控制器**线程基类**——用户自定义控制器继承本类，只重写
`compute_joint_targets() -> q` 一个方法即可。

- `__init__(channel, manipulator, jointTargets=None, ...DLS 参数)`：
  构造即装配——读取关节范围（`_q_min/_q_max`）与初始目标角，自动实例化
  `ServoActuator` 列表并 `channel.attach` 注册
- `run()`（冻结）：`wait_snapshot -> compute_joint_targets() ->
  范围截断 -> 写舵机缓冲 -> mark_done` 循环
- `set_target_joints(q)` / `set_target_ee(position)`：关节/末端目标注入
- 默认控制律：关节模式透传；末端模式每帧一次 DLS 雅可比迭代
  （`_ee_dls_step`，雅可比取自快照 `sensor.eeJacobian`，步长限幅
  `eeMaxStep` 保证稳定）

### 自定义控制器（最小示例）

继承基类、只重写控制函数；在 `sim.wait_ready()` 后构造并 `start()`：

```python
from src import (AerialManipulator, Environment, Manipulator, Multirotor,
                 MujocoSimulation, MultirotorController, ManipulatorController)

class MyDroneController(MultirotorController):
    def control(self):                          # 唯一允许重写的方法
        e = self._target_position - self.sensor.dronePosition
        thrust = self._mass * (self._gravity + 3.0 * e[2] - 2.5 * self.sensor.droneVelocity[2])
        torque = -4.0 * self.sensor.droneOrientation - 0.3 * self.sensor.droneAngularVelocity
        u = self._alloc_pinv @ [thrust, *torque]    # 复用基类混控矩阵
        for a, v in zip(self.actuators, u):
            a.set_thrust(float(v))

class MyArmController(ManipulatorController):
    def compute_joint_targets(self):            # 唯一允许重写的方法
        return [0.3, -0.2, 0.3]

env = Environment()
uam = AerialManipulator(Multirotor(), Manipulator(), compileModel=False)
sim = MujocoSimulation(shutdown, env, uam, useViewer=False)
sim.start(); sim.wait_ready()
drone = MyDroneController(sim.drone_channel, sim.uam.multirotor, targetPosition=[0, 0, 1.8])
arm = MyArmController(sim.arm_channel, sim.uam.manipulator)
drone.start(); arm.start()
sim.start_physics()
```

**语法级冻结**：两个基类通过 `__init_subclass__` 在类创建时检查子类命名空间，
重写除控制函数外的任何基类方法（`__init__` / `run` / 目标注入 / 内部子步骤）
都会当场抛出 `TypeError`——从语法上保证仿真协议链路不被意外改写。

### `MujocoSimulation.py` — class `MujocoSimulation(threading.Thread)`

仿真整体打包。**输入为外部构造好的 `Environment` 与未编译
`AerialManipulator`（`compileModel=False`）两大实例**，本线程把机器人
挂进场景统一编译（唯一编译入口是 `Environment.attach_robot()` ->
`Environment.compile()`），**不接收 Controller 对象**。构造参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `shutdownEvent` | 必填 | 全局退出信号 |
| `environment` | 必填 | `Environment` 实例（场景） |
| `uam` | 必填 | `AerialManipulator` 实例（`compileModel=False`；纯多旋翼构型时 `arm_channel` 为 None） |
| `telemetryBuffer` | None | 遥测输出（供 GCS） |
| `fixedRotorThrust` | None | 固定推力开环 [N]；None 且无订阅者时用 `hover_thrust()` |
| `useViewer` | True | 是否打开 passive viewer（False 为无头） |
| `realTimeFactor` | 1.0 | 实时因子 |
| `telemetryTarget` | None | (host, port) UDP+JSON 跨进程遥测 |
| `telemetryRateHz` | 100 | 遥测发送频率上限 |

握手协议：`wait_ready(timeout)`（编译完成、通道暴露、物理停放）→
外部构造并启动控制器线程 → `start_physics()` 放行。

主循环 `_frame()`（每帧 = 1 个物理步 1 ms）：

```text
sensor = multirotor.get_state()
drone_channel 有订阅者: publish(sensor) -> wait_done -> flush 旋翼缓冲
                        无订阅者: set_actuator(固定/悬停推力)（开环）
arm_channel 有订阅者:   publish(manipulator.get_state()) -> wait_done -> flush 舵机缓冲
-> telemetry（buffer / UDP）
-> env.step() -> viewer.sync() -> 按 RTF 调速
```

退出时 `finally` 关闭两条通道（控制器线程的 `wait_snapshot` 收到
`None` 自行退出）并停止遥测发布器。无 mujoco 时降级为 clock-only 空转。

## 5. MJCF 命名约定（组件自省所依赖）

| 文件 | 约定 |
|---|---|
| multirotor.xml | body `drone`；freejoint `drone_free`；site `imu_site`、`manipulator_mount`、`rotorN_site`；actuator `rotor1..6`（gear 含 ±0.02 反扭矩，奇偶交替）；sensor `gyro/accel/mag/drone_pos/drone_quat/mount_pos/mount_quat` |
| Manipulator.xml | 关节 `joint1/2/3`（hinge）；同名 `<position>` 舵机；site `ee_site`；sensor `jointposN/jointvelN/ee_pos/ee_quat`；无场景元素 |
| environment.xml | site `uam_spawn`；geom `floor`、`obstacle_*`（前缀发现）、`landing_pad` |

注意：MJCF 无 `<barometer>` 元素，气压高度由 `drone_pos` 的 Z 换算。

## 6. 入口（main.py）

```text
python main.py                          默认：位置闭环悬停于出生点 + viewer
python main.py --no-viewer              纯无头运行
python main.py --fixed-thrust 3.5       固定推力开环（不启动控制器线程）
python main.py --pos-target 0 0 2.0     位置闭环（MultirotorController 线程）
python main.py --joint-targets 0.4 0.0  机械臂关节目标角
python main.py --ee-target 0.3 0 1.2    末端位置闭环（ManipulatorController 线程）
python main.py --no-arm                 纯多旋翼构型
python main.py --env <场景.xml> --rtf 1.0
```

调用链：`argparse -> Event + TelemetryBuffer -> Environment + 组件组装
AerialManipulator -> MujocoSimulation(env, uam).start() -> sim.wait_ready()
-> 构造/启动控制器线程 -> sim.start_physics() -> 主线程等待 ->
退出时 shutdown + join（仿真线程 + 控制器线程）`。

## 7. 测试

| 文件 | 覆盖 |
|---|---|
| tests/test_aerial_manipulator.py | 机器人组合、名称前缀、通道驱动悬停平衡、臂-平台耦合（独立模式） |
| tests/test_environment.py | 场景组合、出生位姿、障碍物自省、通道驱动悬停、下落接触交互 |
| tests/test_simulation.py | 三线程无头运行：闭环悬停/爬升、开环下落、仿真与控制器线程退出 |
| tests/test_controllers.py | 混控构建、位置闭环、关节/末端（DLS）控制、三线程集成、用户自定义子类、冻结保护 |

运行方式：`.venv/Scripts/python tests/<文件名>`（项目虚拟环境在 `.venv/`）。

## 8. 已知限制

- 无头模式每帧 1 个物理步 + 帧同步往返的开销使实际速率约为 0.6~0.7 倍
  实时（Windows 睡眠粒度）；需要更高 RTF 时可改为每帧多物理步
- 帧同步邮箱为单订阅者设计：每条通道同时只能挂一个控制器线程
- README.md 仍描述更早的 MVP 结构，待更新
