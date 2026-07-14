# MuJoCo-Mavlink-Bridge

这是一个面向多旋翼平台的 MuJoCo 仿真框架，核心包含：

- **MuJoCo 仿真线程**：加载 MJCF 模型，运行物理仿真，提取状态。
- **无人机控制器接口**：`DroneControllerInterface` 抽象基类，供用户实现自己的控制器。
- **地面站（GCS）**：Tkinter 窗口，实时显示位置/姿态曲线和目标输入框。

> 注意：仓库名包含 `Mavlink`，但当前代码中尚未接入 `pymavlink` 或真实 MAVLink 通信层。现阶段是一个 MuJoCo 仿真与控制器接口框架。

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

只运行仿真，不打开地面站窗口：

```powershell
python Main.py --nogui
```

指定其他模型：

```powershell
python Main.py --model models/Drone.xml
```

当前默认控制器是 `OpenLoopFixedThrustController`，它会给 6 个旋翼持续输出固定推力（默认 `2.9 N`），用于验证 MuJoCo、控制器接口和 actuator 写入链路。

## 目录结构

```text
.
├── Main.py                         # 程序入口
├── requirements.txt                # Python 依赖
├── docs/
│   └── architecture.md             # 架构和调用链说明
├── include/
│   ├── Messages.py                 # 传感器数据结构
│   ├── PIDController.py            # 通用向量 PID（供控制器复用）
│   └── Telemetry.py                # 地面站遥测缓存
├── src/
│   ├── MujocoSimulation.py         # MuJoCo 仿真线程
│   ├── Robot.py                    # 从 MuJoCo 提取无人机状态
│   ├── GroundControlStation.py     # Tkinter 地面站与曲线显示
│   └── drone/
│       └── DroneControllerInterface.py  # 无人机控制器抽象接口
├── cython_modules/
│   └── fast_mujoco_utils.py        # MuJoCo 关节数据提取的 Python fallback
└── models/
    ├── common_uam.xml              # 默认六旋翼简化模型
    ├── Drone.xml                   # 简化六旋翼模型
    ├── UAM_1cable.xml              # 占位模型
    ├── UAM_2cables.xml
    └── UAM_4cables.xml
```

## 运行链路

```text
Main.py
  -> 创建 shutdownEvent 和 TelemetryBuffer
  -> 创建 DroneControllerInterface 实例
  -> 创建并启动 MujocoSimulation
  -> 创建 GCS 地面站窗口
```

仿真循环中，`MujocoSimulation` 会执行：

1. 从 MuJoCo `qpos`/`qvel` 读取无人机位置、速度、姿态和角速度。
2. 封装为 `SensorData`，通过 `setSensorData()` 推给 `DroneControllerInterface`。
3. 将遥测数据追加到 `TelemetryBuffer`，供 GCS 绘图。
4. 从控制器读取 `DroneControlInput.u`。
5. 按 `rotor1`~`rotor6` 写入 MuJoCo actuator。
6. 调用 `mujoco.mj_step()` 推进仿真。

GCS 当前以 `taskManager=None` 创建，因此左侧输入框中的目标位置/姿态只会影响图中的虚线目标，不会发送到默认控制器。

## 核心模块说明

### `Main.py`

程序入口。负责解析命令行参数、创建共享退出事件、遥测缓存、控制器和 MuJoCo 仿真线程。

### `src/MujocoSimulation.py`

MuJoCo 仿真线程。主要职责：

- 加载 MJCF 模型。
- 启动 MuJoCo passive viewer。
- 每帧提取状态并分发给控制器。
- 将控制器输出写入模型 actuator。
- 向 `TelemetryBuffer` 写入位置和姿态数据。
- 调用 `mujoco.mj_step()` 推进仿真。

如果未安装 `mujoco`，该类会进入 clock-only sensor loop，用零状态维持控制器和遥测链路，方便在无 MuJoCo 环境时调试框架接口。

### `src/Robot.py`

从 MuJoCo `model`/`data` 中提取无人机状态，并将控制器输出写入 `data.ctrl`。

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

### `src/GroundControlStation.py`

Tkinter 地面站。界面包含：

- 位置目标输入：`X/Y/Z`
- 姿态目标输入：`Roll/Pitch/Yaw`
- 实际值与目标值曲线
- 遥测清空按钮

## 扩展一个自己的无人机控制器

可以参考 `OpenLoopFixedThrustController`。最小结构如下：

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

## 模型约定

当前默认模型 `models/common_uam.xml` 是一个轻量级六旋翼 MJCF 模型，包含：

- 一个名为 `drone` 的自由体。
- 六个 rotor site：`rotor1_site` 到 `rotor6_site`。
- 六个 motor actuator：`rotor1` 到 `rotor6`。

`MujocoSimulation` 按 `rotor1`~`rotor6` 名称写入控制值。如果模型中缺少某个 actuator，该项会被静默跳过。
