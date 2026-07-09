# MuJoCo-Mavlink-Bridge

这是一个面向空中机械臂/多旋翼平台的 MuJoCo 仿真框架。当前代码主要完成了：

- 通过 `Main.py` 启动 MuJoCo 仿真线程、无人机控制器线程和地面站窗口。
- 将 MuJoCo 中的位姿、速度和欧拉角反馈发送给 `DroneControllerInterface` 控制器。
- 将控制器输出的六路旋翼推力写入 MuJoCo actuator：`rotor1` 到 `rotor6`。
- 在 Tkinter 地面站中实时绘制位置和姿态的实际值与目标值。
- 保留级联 PID、任务管理器、机械臂防摆控制等模块，便于后续接入闭环控制链路。

> 注意：仓库名包含 `Mavlink`，但当前代码中尚未接入 `pymavlink` 或真实 MAVLink 通信层。现阶段更准确地说，它是一个 MuJoCo 仿真与控制器接口框架。

## 环境要求

- Python 3.10 或更高版本
- MuJoCo Python 包
- GLFW
- NumPy
- Matplotlib
- Tkinter（通常随 Python 安装；某些 Linux 发行版需要单独安装）

安装依赖：

```powershell
pip install -r requirements.txt
```

`requirements.txt` 当前包含：

```text
numpy>=1.24
mujoco>=3.2
glfw>=2.7
matplotlib>=3.8
```

## 快速运行

启动默认仿真和地面站：

```powershell
python Main.py
```

默认入口会加载：

```text
models/common_uam.xml
```

如果只运行仿真线程，不打开地面站窗口：

```powershell
python Main.py --nogui
```

指定其他模型：

```powershell
python Main.py --model models/Drone.xml
```

当前默认控制器是 `OpenLoopFixedThrustController`，它会给 6 个旋翼持续输出固定推力，默认每个旋翼 `3 N`。该控制器不读取地面站目标值来闭环跟踪，只用于验证 MuJoCo、控制器接口和 actuator 写入链路。

## 目录结构

```text
.
├── Main.py                         # 当前程序入口
├── requirements.txt                # Python 依赖
├── docs/
│   └── architecture.md             # 架构和调用链说明
├── include/
│   ├── Messages.py                 # 命令、传感器、控制消息数据结构
│   ├── PIDController.py            # 通用向量 PID
│   └── Telemetry.py                # 地面站遥测缓存
├── src/
│   ├── MujocoSimulation.py         # MuJoCo 仿真线程和数据桥接
│   ├── GroundControlStation.py     # Tkinter 地面站与曲线显示
│   ├── TaskManager.py              # GCS 命令到控制目标的转换
│   ├── drone/
│   │   ├── DroneControllerInterface.py
│   │   ├── CascadedPIDDroneController.py
│   │   └── droneControllers/       # 旧版坐标/姿态级联控制器
│   └── manipulator/
│       └── SwayControl.py          # 双臂/关节防摆控制
├── cython_modules/
│   └── fast_mujoco_utils.py        # MuJoCo 关节数据提取的 Python fallback
└── models/
    ├── common_uam.xml              # 当前默认六旋翼简化模型
    ├── Drone.xml                   # 简化六旋翼模型
    ├── UAM_1cable.xml              # include common_uam.xml 的占位模型
    ├── UAM_2cables.xml
    ├── UAM_4cables.xml
    └── Manipulator.xml
```

## 当前运行链路

当前 `Main.py` 实际运行的链路如下：

```text
Main.py
  -> 创建 shutdownEvent 和 TelemetryBuffer
  -> 创建 DroneControllerInterface 实例
  -> 创建并启动 MujocoSimulation
  -> 创建 GCS 地面站窗口
```

仿真循环中，`MujocoSimulation` 会执行：

1. 从 MuJoCo `qpos`/`qvel` 读取无人机位置、速度、姿态和角速度。
2. 写入 `SensorData`，并通过 `setSensorData()` 传给无人机控制器。
3. 将遥测数据追加到 `TelemetryBuffer`，供 GCS 绘图。
4. 从控制器读取 `DroneControlInput.u`。
5. 按顺序映射为 `rotor1` 到 `rotor6` 的 actuator 控制量。
6. 调用 `mujoco.mj_step()` 推进仿真。

GCS 当前以 `taskManager=None` 创建，因此左侧输入框中的目标位置/姿态只会影响图中的虚线目标，不会发送到默认控制器。

更详细的类图和时序图见：

```text
docs/architecture.md
```

## 核心模块说明

### `Main.py`

程序入口。负责解析命令行参数、创建共享退出事件、遥测缓存、控制器和 MuJoCo 仿真线程。

默认使用：

```python
OpenLoopFixedThrustController
```

它继承自 `DroneControllerInterface`，输出固定长度为 6 的 `DroneControlInput.u`。

### `src/MujocoSimulation.py`

MuJoCo 仿真线程。主要职责：

- 加载 MJCF 模型。
- 启动 MuJoCo passive viewer。
- 将 MuJoCo 状态转换成 `SensorData`。
- 将控制器输出写入模型 actuator。
- 向 `TelemetryBuffer` 写入位置和姿态数据。
- 可选接入 `SwayControl` 的关节控制量。

如果未安装 `mujoco`，该类会进入 clock-only sensor loop，用零状态维持控制器和遥测链路，方便在缺少 MuJoCo 环境时调试框架接口。

### `src/drone/DroneControllerInterface.py`

无人机控制器的线程化抽象接口。自定义控制器应继承该类，并实现：

```python
def run(self) -> None:
    ...

def computeControl(self) -> DroneControlInput:
    ...
```

控制器可通过以下方法读取和写入状态：

- `getFeedback()`：读取位置、姿态、速度、角速度反馈。
- `getTargetPose()`：读取目标位置和姿态。
- `setControlInput()`：写入控制输出。
- `waitForUpdate()`：等待仿真线程送来新传感器数据。

控制输出 `DroneControlInput.u` 会被解释为旋翼推力数组：

```text
u[0] -> rotor1
u[1] -> rotor2
...
u[5] -> rotor6
```

### `src/drone/CascadedPIDDroneController.py`

新版级联 PID 控制器，已经实现但当前没有在 `Main.py` 中默认启用。它包含：

- 位置环：位置误差生成期望速度。
- 速度环：速度误差生成期望加速度。
- 姿态目标计算：由水平加速度和 yaw 目标生成 roll/pitch/yaw 目标。
- 姿态环：姿态误差生成期望角速度。
- 角速度环：角速度误差生成 roll/pitch/yaw 控制量。
- Hex-X 混控：输出 6 个旋翼推力。

### `src/GroundControlStation.py`

Tkinter 地面站。界面包含：

- 位置目标输入：`X/Y/Z`
- 姿态目标输入：`Roll/Pitch/Yaw`
- 实际值与目标值曲线
- 遥测清空按钮

当构造函数传入 `taskManager` 时，按钮会发送 `Command("setpoint")`；当前 `Main.py` 传入的是 `None`，所以按钮只改变图上的目标参考线。

### `src/TaskManager.py`

任务管理线程。用于将 GCS 的高层命令转换成 `ControlMessage`，支持：

- `takeoff`
- `land`
- `goto`
- `setpoint`
- `grasp`
- `release`
- `sway`

该模块属于保留的闭环控制链路，目前没有被默认入口接入。

### `src/manipulator/SwayControl.py`

机械臂/关节防摆控制线程。它读取 `RobotSensorData` 中的关节位置、速度和 `qCombinationCase`，生成 8 个关节电机控制量。当前定义的关节名包括：

```text
left_shoulder_roll
left_shoulder_pitch
left_elbow_roll
left_elbow_pitch
right_shoulder_roll
right_shoulder_pitch
right_elbow_roll
right_elbow_pitch
```

要启用该控制器，模型中需要存在对应名称的 actuator。

## 模型约定

当前默认模型 `models/common_uam.xml` 是一个轻量级六旋翼 MJCF 模型，包含：

- 一个名为 `drone` 的自由体。
- 六个 rotor site：`rotor1_site` 到 `rotor6_site`。
- 六个 motor actuator：`rotor1` 到 `rotor6`。

`MujocoSimulation._applyControl()` 会按 actuator 名称写入控制值。如果模型中缺少某个 actuator，该项会被跳过。

`UAM_1cable.xml`、`UAM_2cables.xml`、`UAM_4cables.xml` 等文件目前是轻量占位/包含式模型，完整空中机械臂模型和网格资源需要后续替换。

## 扩展一个自己的无人机控制器

可以参考 `OpenLoopFixedThrustController` 或 `CascadedPIDDroneController`。最小结构如下：

```python
import threading
import numpy as np

from src.drone import DroneControllerInterface, DroneControlInput


class MyController(DroneControllerInterface):
    def __init__(self, shutdownEvent: threading.Event):
        super().__init__(shutdownEvent)

    def run(self) -> None:
        while not self._shutdownEvent.is_set():
            self.waitForUpdate(timeout=0.05)
            self.setControlInput(self.computeControl())

    def computeControl(self) -> DroneControlInput:
        feedback = self.getFeedback()
        thrust = np.full(6, 3.0, dtype=float)
        return DroneControlInput(u=thrust)
```

然后在外部调用 `main()` 时传入工厂函数：

```python
import threading

from Main import main
from my_controller import MyController


def build_controller(shutdownEvent: threading.Event):
    return MyController(shutdownEvent)


main(build_controller)
```

## 当前状态与后续方向

当前已经可用的部分：

- MuJoCo 仿真启动
- 六旋翼 actuator 写入
- 控制器线程接口
- 实时遥测绘图
- 固定推力开环测试
- 级联 PID 控制器代码
- 任务管理器和机械臂防摆控制代码骨架

仍待接入或完善的部分：

- MAVLink 通信层
- GCS、`TaskManager` 与新版 `DroneControllerInterface` 的完整目标链路
- 级联 PID 控制器的默认运行配置和参数调试
- 完整 UAM/机械臂 MJCF 模型、mesh 和 actuator 定义
- 自动化测试与控制效果验证
