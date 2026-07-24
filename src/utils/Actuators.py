"""执行器缓冲：控制器线程写指令，仿真线程帧边界统一落盘 mjData。

控制器线程只调用 ``set_thrust`` / ``set_target`` 修改**缓冲值**
（纯 Python float，无 mjData 访问，无数据竞争）；``flush()`` 把缓冲值
写入 ``data.ctrl``，**只能由仿真线程调用**——全进程只有仿真线程访问
mjData。
"""

from __future__ import annotations

import numpy as np


class _BufferedActuator:
    """单通道执行器缓冲基类：范围截断 + 帧边界落盘。"""

    def __init__(self, data, act_id: int, ctrl_range, initial: float, name: str = ""):
        self._data = data
        self._act_id = int(act_id)
        self._lo = float(ctrl_range[0])
        self._hi = float(ctrl_range[1])
        self._cmd = float(np.clip(initial, self._lo, self._hi))
        self.name = name

    @property
    def command(self) -> float:
        """当前缓冲指令（截断后）。"""
        return self._cmd

    def _set(self, value: float) -> None:
        self._cmd = float(np.clip(value, self._lo, self._hi))

    def flush(self) -> None:
        """把缓冲指令写入 data.ctrl。**只能由仿真线程调用。**"""
        self._data.ctrl[self._act_id] = self._cmd


class RotorActuator(_BufferedActuator):
    """旋翼执行器缓冲（指令 = 推力 [N]）。"""

    def set_thrust(self, thrust: float) -> None:
        """设置推力指令 [N]（自动按 ctrlrange 截断，帧边界落盘）。"""
        self._set(thrust)


class ServoActuator(_BufferedActuator):
    """舵机执行器缓冲（指令 = 关节目标角 [rad]，位置伺服）。"""

    def set_target(self, angle: float) -> None:
        """设置目标角指令 [rad]（自动按关节范围截断，帧边界落盘）。"""
        self._set(angle)
