"""帧同步通信：仿真线程 <-> 控制器线程。

三个线程模型：

- **仿真线程**（MujocoSimulation）：唯一访问 mjData 的线程，逐帧推进物理；
- **多旋翼控制器线程** / **机械臂控制器线程**：各自独立线程，只读写
  通道对象，绝不触碰 mjData。

通信分两个方向，全部封装在 :class:`ControllerChannel` 里：

- 仿真 -> 控制器：:class:`FrameMailbox` 帧同步邮箱，每帧发布一份传感
  快照（当帧拷贝），并阻塞等待控制器回执——控制频率与物理帧严格同步；
- 控制器 -> 仿真：执行器缓冲（见 ``Actuators.py``），控制器把指令写进
  缓冲，仿真线程在帧边界统一 ``flush()`` 进 mjData。

Simulation 不接收 Controller 对象：通道由仿真线程在编译后创建并暴露
（``sim.drone_channel`` / ``sim.arm_channel``），控制器在线程外构造时
通过 ``channel.attach(actuators)`` 注册自己。
"""

from __future__ import annotations

import threading


class FrameMailbox:
    """帧同步邮箱（单发布者/单订阅者）。

    仿真线程（发布者）：
        ``publish(snapshot)`` 发布第 N 帧 -> ``wait_done()`` 阻塞到回执。
    控制器线程（订阅者）：
        ``wait_snapshot(last_frame)`` 阻塞到新帧 -> 处理 -> ``mark_done(frame)``。

    ``close()`` 后 ``wait_snapshot`` 返回 ``(None, last_frame)``，控制器
    线程据此退出。
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._snapshot = None
        self._frame = 0
        self._done_frame = 0
        self._closed = False

    # ---------- 发布者（仿真线程） ----------

    def publish(self, snapshot) -> int:
        """发布一帧快照，返回帧号。"""
        with self._cond:
            self._frame += 1
            self._snapshot = snapshot
            self._cond.notify_all()
            return self._frame

    def wait_done(self, shutdown: threading.Event | None = None) -> bool:
        """阻塞直到最新帧被回执；``shutdown`` 置位或邮箱关闭时提前返回 False。"""
        with self._cond:
            while self._done_frame < self._frame and not self._closed:
                if shutdown is not None and shutdown.is_set():
                    return False
                self._cond.wait(0.05)
            return not self._closed

    def close(self) -> None:
        """关闭邮箱：唤醒所有等待方，订阅者下一轮 ``wait_snapshot`` 收到 None。"""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # ---------- 订阅者（控制器线程） ----------

    def wait_snapshot(self, last_frame: int):
        """阻塞直到出现比 ``last_frame`` 新的帧，返回 ``(snapshot, frame)``；
        邮箱关闭且无新帧时返回 ``(None, last_frame)``。"""
        with self._cond:
            while (self._frame <= last_frame or self._snapshot is None) and not self._closed:
                self._cond.wait()
            if self._frame <= last_frame or self._snapshot is None:
                return None, last_frame
            return self._snapshot, self._frame

    def mark_done(self, frame: int) -> None:
        """回执：第 ``frame`` 帧已处理完（控制指令已写入执行器缓冲）。"""
        with self._cond:
            self._done_frame = max(self._done_frame, frame)
            self._cond.notify_all()


class ControllerChannel:
    """仿真线程与单个控制器线程之间的通道（邮箱 + 执行器缓冲注册表）。

    由仿真线程在模型编译后创建并暴露；控制器（在线程外）构造时把执行器
    缓冲列表注册进来——注册即视为订阅，仿真线程从下一帧开始走同步协议。
    """

    def __init__(self) -> None:
        self.mailbox = FrameMailbox()
        self._actuators: list = []
        self.has_subscriber = False

    def attach(self, actuators: list) -> None:
        """控制器注册执行器缓冲列表（线程外调用，物理启动前完成）。"""
        self._actuators = list(actuators)
        self.has_subscriber = True

    def flush(self) -> None:
        """把全部执行器缓冲写入 mjData。**只能由仿真线程在帧边界调用。**"""
        for actuator in self._actuators:
            actuator.flush()

    def close(self) -> None:
        """关闭通道（仿真退出时调用，控制器线程随之退出）。"""
        self.mailbox.close()
