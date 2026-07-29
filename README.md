# UAMSim — MuJoCo 空中机械臂仿真器

面向多旋翼 + 机械臂（UAM, Unmanned Aerial Manipulator）的 MuJoCo 仿真与控制框架。采用**三线程实时架构**（仿真线程 + 多旋翼控制器线程 + 机械臂控制器线程），通过帧同步机制保证控制频率与物理帧严格对齐。

---

## 安装

```bash
# 基础安装
pip install -e .

# 带开发依赖（lint / format / test）
pip install -e ".[dev]"
```

### 系统要求

- Python >= 3.10
- numpy >= 1.24
- mujoco >= 3.2
- glfw >= 2.7
- matplotlib >= 3.8

---

## 30 秒快速开始

```python
import threading
import numpy as np
import UAMSim as ua


class HoverController(ua.MultirotorController):
    """自定义控制器：只需重写 controller() 方法。"""

    def controller(self, feedback, target_pos, kp, hover):
        err = target_pos - feedback.dronePosition
        thrust = hover + kp * err[2]
        return np.full(self._multirotor.n_rotors, thrust)


# 1. 加载场景与机器人模型
shutdown = threading.Event()
env = ua.Environment("path/to/environment.xml")
multirotor = ua.Multirotor("path/to/multirotor.xml")

# 2. 启动仿真线程（自动组合、编译、创建帧同步通道）
sim = ua.MujocoSimulation(shutdown, env, multirotor, use_viewer=True)
sim.start()
sim.wait_ready()          # 等待编译完成、通道就绪

# 3. 构造并启动控制器（自动注册到帧同步通道）
ctrl = HoverController(
    sim.uam.multirotor, sim.drone_channel, shutdown,
    target_pos=np.array([0.0, 0.0, 1.5]),
    kp=4.0,
    hover=sim.uam.hover_thrust(),
)
ctrl.start()

# 4. 放行物理推进
sim.start_physics()

# 5. 运行一段时间后退出
# shutdown.set()   # 设置退出标志，仿真与控制器自动停止
```

完整示例见 [`examples/demo.py`](examples/demo.py)。

---

## 运行示例

```bash
# 纯多旋翼悬停演示（3 秒，GUI）
python examples/demo.py

# 完整集成演示：级联 PID + 机械臂 ±1 rad 摆动（9 秒）
python examples/demo.py --full

# 无头模式（不开 viewer，5× 加速）
python examples/demo.py --full --headless
```

---

## 包结构

```text
UAMSim/
├── __init__.py                    # 公共 API 导出
├── simulation/
│   ├── environment.py             # Environment：场景组合器
│   ├── multirotor.py              # Multirotor：多旋翼平台组件
│   ├── manipulator.py             # Manipulator：机械臂组件
│   ├── robot.py                   # Robot：已挂载机器人的视图句柄
│   └── mujoco_simulation.py       # MujocoSimulation：仿真线程（三线程核心）
├── controllers/
│   ├── multirotor_controller.py   # MultirotorController：多旋翼控制器基类
│   └── manipulator_controller.py  # ManipulatorController：机械臂控制器基类
└── utils/
    ├── actuators.py               # RotorActuator / ServoActuator 执行器缓冲
    ├── frame_sync.py              # FrameMailbox / ControllerChannel 帧同步
    ├── perception_bus.py          # SensorSnapshot / PerceptionSource / UDP 感知通道
    ├── telemetry.py               # TelemetryBuffer 遥测缓存
    └── telemetry_publisher.py     # TelemetryPublisher UDP 遥测发布

examples/
├── demo.py                        # 综合演示脚本（悬停 / 级联 PID + 机械臂）
└── models/
    ├── environment.xml            # 场景 MJCF（地板、灯光、障碍物）
    ├── multirotor.xml             # 六旋翼平台 MJCF
    └── Manipulator.xml            # 三自由度机械臂 MJCF
```

---

## 核心设计

### 三线程模型

| 线程 | 职责 | 数据访问 |
|------|------|----------|
| **仿真线程** (`MujocoSimulation`) | 唯一访问 `mjData`，逐帧推进物理、发布传感快照、flush 执行器缓冲 | 读写 `mjData` |
| **多旋翼控制器线程** | 接收快照 → 计算推力 → 写入 `RotorActuator` 缓冲 | 只读快照，不写 `mjData` |
| **机械臂控制器线程** | 接收快照 → 计算关节目标角 → 写入 `ServoActuator` 缓冲 | 只读快照，不写 `mjData` |

### 自定义控制器

唯一需要重写的是 `controller()` 方法。构造参数通过 `**kwargs` 逐帧透传：

```python
class MyDroneCtrl(ua.MultirotorController):
    def controller(self, feedback, target, kp):
        err = target - feedback.dronePosition
        return self.uam.hover_thrust() + kp * err[2]

ctrl = MyDroneCtrl(sim.uam.multirotor, sim.drone_channel,
                   target=[0, 0, 1.5], kp=2.0)
ctrl.start()
```

运行期间可随时更新目标参数（下一帧生效）：

```python
ctrl.set_target(target=[0, 0, 2.0], kp=5.0)
```

### 构型灵活

- **纯多旋翼**：`MujocoSimulation(..., multirotor, None)`
- **多旋翼 + 机械臂**：`MujocoSimulation(..., multirotor, manipulator)`
- **自定义模型**：传入自己的 MJCF 路径即可

---

## 详细文档

使用指南、MJCF 命名约定、API 参考等详见 [`docs/DOCUMENT.md`](docs/DOCUMENT.md)。

---

## 许可证

MIT
