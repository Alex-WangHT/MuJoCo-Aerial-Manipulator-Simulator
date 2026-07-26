# uamsim — MuJoCo 空中机械臂仿真器

可安装的 Python 包，面向多旋翼 + 机械臂（UAM）的 MuJoCo 仿真与控制框架。

## 安装

```bash
pip install -e .
```

或带开发依赖：

```bash
pip install -e ".[dev]"
```

## 依赖

- Python >= 3.10
- numpy >= 1.24
- mujoco >= 3.2
- glfw >= 2.7
- matplotlib >= 3.8

## 快速开始

```python
import threading
import numpy as np
import uamsim as ua


class HoverController(ua.MultirotorController):
    """自定义控制器：只需重写 controller() 方法。"""

    def controller(self, feedback, target_pos, hover):
        err = target_pos - feedback.dronePosition
        thrust = hover + 4.0 * err[2]
        return np.full(self._multirotor.n_rotors, thrust)


# 1. 创建场景
env = ua.Environment()

# 2. 启动仿真线程
shutdown = threading.Event()
sim = ua.MujocoSimulation(
    shutdown, env, ua.Multirotor(), None,
    use_viewer=True, real_time_factor=1.0,
)
sim.start()
sim.wait_ready()

# 3. 创建并启动控制器
ctrl = HoverController(
    sim.uam.multirotor, sim.drone_channel, shutdown,
    target_pos=np.array([0.0, 0.0, 1.5]),
    hover=sim.uam.hover_thrust(),
)
ctrl.start()

# 4. 放行物理
sim.start_physics()
```

完整示例见 `examples/hover_example.py`。

## 包结构

```text
uamsim/
├── __init__.py                    # 公共 API 导出
├── simulation/
│   ├── environment.py             # Environment：场景组合器
│   ├── multirotor.py              # Multirotor：多旋翼平台组件
│   ├── manipulator.py             # Manipulator：机械臂组件
│   ├── robot.py                   # Robot：已挂载机器人的视图句柄
│   └── mujoco_simulation.py       # MujocoSimulation：仿真线程
├── controllers/
│   ├── multirotor_controller.py   # 多旋翼控制器线程基类
│   └── manipulator_controller.py  # 机械臂控制器线程基类
├── utils/
│   ├── actuators.py               # RotorActuator / ServoActuator
│   ├── frame_sync.py              # FrameMailbox / ControllerChannel
│   ├── perception_bus.py          # SensorSnapshot / PerceptionSource
│   ├── telemetry.py               # TelemetryBuffer
│   └── telemetry_publisher.py     # TelemetryPublisher
└── models/                        # MJCF 模型文件
    ├── multirotor.xml
    ├── Manipulator.xml
    └── environment.xml
```

## 核心设计

### 三线程模型

- **仿真线程**（`MujocoSimulation`）：唯一访问 `mjData` 的线程，逐帧推进物理
- **多旋翼控制器线程** / **机械臂控制器线程**：各自独立线程，只读写通道对象

### 自定义控制器

唯一需要重写的是 `controller()` 方法：

```python
class MyController(ua.MultirotorController):
    def controller(self, feedback, **params):
        # feedback: SensorSnapshot（含位置、速度、姿态、角速度等）
        # params: 构造时传入的自定义参数
        return np.full(self._multirotor.n_rotors, 3.0)  # 各旋翼推力 [N]
```

自定义参数通过构造函数 kwargs 透传：

```python
ctrl = MyController(
    multirotor, channel, shutdown,
    target_pos=[0, 0, 1.5], kp=4.0, kd=3.0,
)
```

### 构型灵活

- **纯多旋翼**：`MujocoSimulation(..., multirotor, None)`
- **多旋翼 + 机械臂**：`MujocoSimulation(..., multirotor, manipulator)`
- **自定义模型路径**：`Multirotor(multirotor_path="path/to/my_multirotor.xml")`

## 测试

```bash
# 场景组合与交互测试
python tests/test_environment.py

# 机器人整机组合测试
python tests/test_robot.py

# 仿真线程三线程模型测试
python tests/test_simulation.py

# 整机集成测试（级联 PID + 机械臂摆动）
python tests/test_integrated.py           # GUI 模式
python tests/test_integrated.py --headless  # 无头模式
```

## 许可证

MIT
