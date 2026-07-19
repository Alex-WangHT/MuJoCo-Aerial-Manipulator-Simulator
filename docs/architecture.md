# Architecture

> 本文档以当前 `src/` 实际代码为准（develop 分支重构后），描述分层结构、
> 模型组合机制与运行时调用关系。

## 1. 目录结构

```text
.
├── main.py                        # 程序入口（最终调用）
├── requirements.txt               # numpy / mujoco>=3.2 / glfw / matplotlib
├── docs/
│   └── architecture.md            # 本文档
├── models/
│   ├── environment.xml            # 场景：地板/灯光/障碍物/着陆台/uam_spawn
│   ├── multirotor.xml             # 六旋翼平台（纯机器人本体，无场景元素）
│   └── Manipulator.xml            # 二自由度机械臂（子模型）
├── src/
│   ├── MujocoSimulation/          # 仿真包：全部仿真组件
│   │   ├── __init__.py            #   包导出 + quaternionToEuler + _MODELS_DIR
│   │   ├── Messages.py            #   SensorData / ControlInput 数据类
│   │   ├── Telemetry.py           #   TelemetryBuffer（线程安全遥测环形缓冲）
│   │   ├── Multirotor.py          #   平台视图（旋翼/机体状态/IMU/挂载点/旋翼几何）
│   │   ├── Manipulator.py         #   机械臂视图（关节/舵机/末端）
│   │   ├── AerialManipulator.py   #   机器人组合器（平台 + 机械臂）
│   │   ├── Environment.py         #   场景组合器（最顶层）
│   │   ├── MultirotorController.py    # 多旋翼控制器基类（默认串级 PID + 混控）
│   │   ├── ManipulatorController.py   # 机械臂控制器基类（默认关节/末端 DLS）
│   │   └── MujocoSimulation.py    #   仿真线程（仿真整体打包）
│   └── （GroundControlStation.py 已移除；GCS 待用户重写，main.py 始终无界面运行）
└── tests/                         # 四个端到端测试（见第 7 节）
```

包外代码一律从包根导入：`from src.MujocoSimulation import MujocoSimulation`
（`__init__.py` 再导出全部类与数据载体）；包内各文件之间用相对导入
（`from .Environment import Environment`）。

## 2. 分层与调用关系

```text
main.py
  ├── threading.Event              全局退出信号
  ├── TelemetryBuffer              遥测缓冲（仿真线程写，GCS 读）
  ├── MujocoSimulation (thread)    仿真整体
  │     ├── Environment            加载 environment.xml，最顶层组合器
  │     │     └── AerialManipulator (compileModel=False)
  │     │           ├── Multirotor   视图（namespace="uam/"）
  │     │           └── Manipulator  视图（namespace="uam/arm/"）
  │     ├── MultirotorController   bind -> Multirotor；协议 setSensorData/getControlInput
  │     ├── ManipulatorController  bind -> Manipulator；每帧 update()
  │     ├── TelemetryBuffer        每帧追加 (t, position, euler)
  │     └── mujoco.viewer          useViewer=True 时可视化
```

**依赖方向单向向下**：视图只依赖包内 `Messages`；组合器依赖视图；控制器基类
依赖视图（几何/质量自省）；仿真线程依赖组合器与控制器；入口依赖一切。层间只通过
`MjModel`/`MjData` 共享引用和小数据类（SensorData/ControlInput）通信。

## 3. 模型组合机制（mjSpec）

所有组合发生在 **spec 层**，最终只编译**一个** MjModel —— 机械臂与平台、
机器人与场景共享同一套动力学，反作用力/接触天然耦合：

```text
environment.xml (父)
  └── Environment.attach_robot: attach(uam_spec, site="uam_spawn", prefix="uam/")
        multirotor.xml
          └── AerialManipulator: attach(arm_spec, site="manipulator_mount", prefix="arm/")
                Manipulator.xml
```

- 两级名称前缀：`uam/rotor1`、`uam/drone`、`uam/arm/joint1`、`uam/arm/ee_site`
- `AerialManipulator` 两种模式：独立运行（`compileModel=True`，自身编译、
  无前缀）或放入场景（`compileModel=False`，由 Environment 编译后经
  `bind_views(model, data, namespace)` 绑定视图）
- 挂载关系全部声明在 MJCF：机械臂固连位置 = multirotor.xml 中
  `manipulator_mount` site（机体系 `pos="0 0 -0.05"`）；机器人出生位置 =
  environment.xml 中 `uam_spawn` site——Python 中不硬编码坐标
- 场景元素（地板/灯光/障碍物）只属于 environment.xml；机器人 MJCF 保持纯净
- 模型目录常量 `_MODELS_DIR` 定义在包 `__init__.py`（指向仓库根 `models/`），
  组合器经 `from . import _MODELS_DIR` 使用

## 4. 模块职责与公开 API

以下类均位于 `src/MujocoSimulation/` 包内，并从包根再导出。

### 数据载体（`Messages.py` / `Telemetry.py`）

- `SensorData`：timestamp/timestep/位置/速度/欧拉角/角速度
- `ControlInput`：控制器输出，旋翼推力向量 `u` [N]
- `TelemetryBuffer(maxSamples=5000)`：锁保护环形缓冲，`append/snapshot/clear`，
  跨线程（仿真写 → GCS 读）安全

### `Environment.py` — class `Environment`

场景组合器与场景交互查询。

| 方法 | 说明 |
|---|---|
| `__init__(environmentPath=None)` | 加载场景 MJCF（默认 models/environment.xml） |
| `attach_robot(uam, site="uam_spawn", prefix="uam/")` | 挂载未编译机器人并统一编译 |
| `compile()` | 编译场景共享模型，绑定全部机器人视图，刷新障碍物表 |
| `step(n_steps=1)` / `reset()` | 推进 / 重置（mj_resetData + mj_forward） |
| `get_obstacle_positions()` | `obstacle_` 前缀 geom 的世界坐标 |
| `get_contacts()` | 全部接触对 `(geom1, geom2, pos)` |
| `get_robot_contacts(prefix="uam/")` | 只保留涉及指定机器人的接触对 |

属性：`model`、`data`、`robots`、`obstacles`。

### `AerialManipulator.py` — class `AerialManipulator`

机器人（UAM）组合器。

| 方法 | 说明 |
|---|---|
| `__init__(multirotorPath=None, manipulatorPath=None, mountSite="manipulator_mount", armPrefix="arm/", compileModel=True)` | 构建机器人 spec（平台 + attach 机械臂） |
| `compile()` | 独立模式：编译自身 spec 并绑定无前缀视图 |
| `bind_views(model, data, namespace="")` | 绑定视图到（场景共享的）模型 |
| `step(n_steps=1)` / `reset()` | 推进 / 重置（共享模型时 reset 会重置整个场景） |
| `hover_thrust()` | 整机悬停单旋翼推力：总质量 × 9.81 / 旋翼数 |

属性：`spec`、`model`、`data`、`multirotor`、`manipulator`。

### `Multirotor.py` — class `Multirotor`

平台视图（不持有模型，绑定共享 MjModel/MjData）。
类常量：`BODY_NAME="drone"`、`FREEJOINT_NAME="drone_free"`、
`IMU_SITE_NAME="imu_site"`、`MOUNT_SITE_NAME="manipulator_mount"`。

- 构造时自省：扫描 actuator 按 `rotorN` 正则发现旋翼（`rotors` 字典）；
  按约定名查 body/joint/site/sensor（`sensor_adr` + `sensor_dim`）
- `set_thrusts(u)`：按 rotor1..N 顺序写推力 [N]
- `get_state() -> SensorData`：自由关节 qpos/qvel → 位置/速度/欧拉角/角速度
- `get_sensor(key)` / `get_imu()`：gyro/accel（+mag）读数
- `get_mount_pose()`：挂载点世界位姿（位置 + 3×3 旋转矩阵）
- `get_rotor_geometry()`：旋翼机体系坐标 + 单位推力 Z 反扭矩系数
  （取自 gear）+ 推力范围——控制分配/混控所需几何
- 属性：`n_rotors`、`rotor_names`、`body_id`、`model`

### `Manipulator.py` — class `Manipulator`

机械臂视图。类常量：`EE_SITE_NAME="ee_site"`。

- 构造时自省：按命名空间前缀发现非 free 关节（`joints`）；舵机经
  `actuator_trnid` 反查绑定关节（`servos`，不依赖命名）；前缀传感器表
- `set_joint_targets(q)`：写位置舵机目标角 [rad]
- `get_joint_positions()` / `get_joint_velocities()`
- `get_ee_pose()`：末端世界位姿
- `get_sensor(local_key)`：如 `'jointpos1'`、`'ee_pos'`
- 属性：`n_joints`、`joint_names`、`sensor_names`、`ee_site_id`、`model`、`data`

### `MultirotorController.py` — class `MultirotorController`（基类）

多旋翼控制器**基类**——用户自定义控制器继承本类，只重写
`compute_control(sensor) -> u` 一个方法即可；协议、装配、混控与
推力截断全部由基类完成。基类提供：

- `bind(multirotor)`：自省旋翼几何/质量/推力范围，构建 4×N 分配矩阵伪逆
  （`_alloc_pinv`、`_mass`、`_gravity`、`_ctrl_min/_ctrl_max` 供子类复用）；
  由仿真线程组装时自动回调（两段式构造）
- droneController 协议：`setSensorData`（调用控制律并截断推力）/
  `getControlInput() -> ControlInput`
- `set_target_position(pos, yaw=None)`：目标注入点（供 GCS/MAVLink）
- 默认控制律 `compute_control`：串级 PID——位置环 PID → 期望加速度 →
  小角近似分配 roll/pitch → 姿态环 PID → 力矩 → 伪逆混控

### `ManipulatorController.py` — class `ManipulatorController`（基类）

机械臂控制器**基类**——用户自定义控制器继承本类，只重写
`compute_joint_targets() -> q` 一个方法即可；装配、舵机写入与
关节范围截断全部由基类完成。基类提供：

- `bind(manipulator)`：读取关节 dof 地址与关节范围（`_dof_adrs`、
  `_q_min/_q_max` 供子类复用）；仿真线程组装时自动回调
- `set_target_joints(q)` / `set_target_ee(position)`：关节/末端目标注入
- `update()`：仿真主循环每帧调用，取控制律结果写舵机（关节范围截断）
- 默认控制律：关节模式透传；末端模式每帧一次 DLS 雅可比迭代
  （`_ee_dls_step`，步长限幅 `eeMaxStep` 保证稳定；增益/阻尼/容差可调）

### 自定义控制器（最小示例）

继承基类、只重写控制函数，注入仿真即可——装配与协议全自动：

```python
from src.MujocoSimulation import MultirotorController, ManipulatorController

class MyDroneController(MultirotorController):
    def compute_control(self, sensor):          # 唯一允许重写的方法
        e = self._target_position - sensor.dronePosition
        thrust = self._mass * (self._gravity + 3.0 * e[2] - 2.5 * sensor.droneVelocity[2])
        torque = -4.0 * sensor.droneOrientation - 0.3 * sensor.droneAngularVelocity
        return self._alloc_pinv @ [thrust, *torque]   # 复用基类混控矩阵

class MyArmController(ManipulatorController):
    def compute_joint_targets(self):            # 唯一允许重写的方法
        return [0.3, -0.2]

sim = MujocoSimulation(shutdown, droneController=MyDroneController([0, 0, 1.8]),
                       manipulatorController=MyArmController(), useViewer=False)
sim.start()   # bind() 由仿真组装时自动回调
```

**语法级冻结**：两个基类通过 `__init_subclass__` 在类创建时检查子类命名空间，
重写除控制函数外的任何基类方法（`__init__` / `bind` / 协议方法 / 目标注入 /
内部子步骤）都会当场抛出 `TypeError`——从语法上保证仿真协议链路不被意外改写：

```python
class BadController(MultirotorController):
    def bind(self, multirotor): ...     # TypeError: 基类方法 ['bind'] 已冻结
```

子类需要保存运行状态时，用类属性或在控制函数内惰性初始化
（`if not hasattr(self, "_t"): self._t = 0.0`）；构造参数
（目标点、增益、DLS 参数）一律经基类构造器传入，无需重写 `__init__`。

### `MujocoSimulation.py` — class `MujocoSimulation(threading.Thread)`

仿真整体打包。构造参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `shutdownEvent` | 必填 | 全局退出信号 |
| `environmentPath` | None | 场景 MJCF（默认 models/environment.xml） |
| `telemetryBuffer` | None | 遥测输出（供 GCS） |
| `droneController` | None | 多旋翼控制器注入点（`bind` 钩子自动回调） |
| `manipulatorController` | None | 机械臂控制器注入点（`bind` + 每帧 `update`） |
| `fixedRotorThrust` | None | 固定推力开环 [N]；None 且无控制器时用 `hover_thrust()` |
| `jointTargets` | None | 机械臂初始关节目标角 [rad]（有臂控制器时忽略） |
| `useViewer` | True | 是否打开 passive viewer（False 为无头） |
| `realTimeFactor` | 1.0 | 实时因子 |

主循环 `_frame()`（每帧 = 1 个物理步 1 ms）：

```text
sensor = multirotor.get_state()
  ├─> droneController.setSensorData(sensor)                        （若有控制器）
  └─> telemetryBuffer.append(t, pos, euler)                      （若有遥测）
u = controller.getControlInput().u  |  fixedRotorThrust  |  hover_thrust()
multirotor.set_thrusts(u) -> manipulatorController.update()? -> env.step()
-> viewer.sync() -> 按 RTF 调速
```

`run()`：组装（含控制器 `bind` 回调）-> 诊断打印（质量/悬停推力/COM 偏移/
障碍物）-> viewer 或无头循环；`stop()`：置退出信号。无 mujoco 时降级为
clock-only 空转（维持控制器/遥测链路）。

## 5. MJCF 命名约定（视图自省所依赖）

| 文件 | 约定 |
|---|---|
| multirotor.xml | body `drone`；freejoint `drone_free`；site `imu_site`、`manipulator_mount`、`rotorN_site`；actuator `rotor1..6`（gear 含 ±0.02 反扭矩，奇偶交替）；sensor `gyro/accel/mag/drone_pos/drone_quat/mount_pos/mount_quat` |
| Manipulator.xml | 关节 `joint1/2`（hinge，range ±1.8）；同名 `<position>` 舵机；site `ee_site`；sensor `jointposN/jointvelN/ee_pos/ee_quat`；无场景元素 |
| environment.xml | site `uam_spawn`；geom `floor`、`obstacle_*`（前缀发现）、`landing_pad` |

注意：MJCF 无 `<barometer>` 元素，气压高度由 `drone_pos` 的 Z 换算。

## 6. 入口（main.py）

```text
python main.py                          默认场景 + 悬停推力 + viewer
python main.py --no-viewer              纯无头运行
python main.py --fixed-thrust 3.5       固定推力开环
python main.py --pos-target 0 0 2.0     位置闭环（MultirotorController）
python main.py --joint-targets 0.4 0.0  机械臂关节目标角
python main.py --ee-target 0.3 0 1.2    末端位置闭环（ManipulatorController）
python main.py --env <场景.xml> --rtf 1.0
```

调用链：`argparse -> Event + TelemetryBuffer (+ 控制器) -> MujocoSimulation.start()
-> 主线程等待 -> 退出时 shutdown + join`。

## 7. 测试

| 文件 | 覆盖 |
|---|---|
| tests/test_aerial_manipulator.py | 机器人组合、名称前缀、悬停平衡、臂-平台耦合（独立模式） |
| tests/test_environment.py | 场景组合、出生位姿、障碍物自省、悬停、下落接触交互 |
| tests/test_simulation.py | 仿真线程无头运行：悬停遥测、固定推力上升/下落、线程退出 |
| tests/test_controllers.py | 混控构建、位置闭环、关节/末端（DLS）控制、线程集成、用户自定义子类 |

运行方式：`.venv/Scripts/python tests/<文件名>`（项目虚拟环境在 `.venv/`）。

## 8. 已知限制

- 无头模式每帧 1 个物理步 + 传感提取的开销使实际速率约为 0.6~0.7 倍实时
  （Windows 睡眠粒度）；需要更高 RTF 时可改为每帧多物理步
- 旧文件已移除：`include/`（Messages/Telemetry 已收进包内）、
  `src/GroundControlStation.py`（GCS 待用户重写）、
  `src/ControlEnvironment.py`、`src/Robot.py`、`models/common_uam.xml`、
  `models/Drone.xml`、`models/UAM_*cables.xml`、`.gitmodules`（PX4 子模块路线已弃）
- README.md 仍描述更早的 MVP 结构，待更新
