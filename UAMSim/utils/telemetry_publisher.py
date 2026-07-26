"""跨进程遥测发布通道（UDP + JSON，独立线程）。

设计目标：**绝不阻塞仿真主循环**。

- 仿真主循环每帧只调用 :meth:`TelemetryPublisher.publish`，内部是
  有界队列的 ``put_nowait``（微秒级，无锁等待）；队列满时丢弃最旧一帧，
  保证新数据始终优先、主循环零等待。
- 序列化（JSON）与 UDP 发送全部在本类的后台守护线程内完成；
  发送失败（对端未监听等）静默忽略——遥测本身允许丢包。
- 协议：每个 UDP 数据报 = 一行 JSON（UTF-8，``\\n`` 结尾），
  接收端用任意语言 ``recvfrom`` 后按行 ``json.loads`` 即可。
"""

from __future__ import annotations

import json
import queue
import socket
import threading


class TelemetryPublisher:
    """把仿真状态经 UDP 发往另一个进程（非阻塞、独立线程）。

    参数:
        host: 目标主机（本机调试用 ``127.0.0.1``）
        port: 目标 UDP 端口
        max_queue: 待发送队列长度上限（满时丢最旧帧，保护主循环）
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9100, max_queue: int = 1000):
        self._addr = (host, int(port))
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=max_queue)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="TelemetryPublisher"
        )
        self.sent = 0
        self.dropped = 0

    # ---------- 生命周期 ----------

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        try:
            self._socket.close()
        except OSError:
            pass

    # ---------- 主循环侧（仿真线程调用） ----------

    def publish(self, packet: dict) -> None:
        """入队一帧遥测，立即返回。**永不阻塞、永不抛异常**。"""
        try:
            payload = (json.dumps(packet, separators=(",", ":")) + "\n").encode("utf-8")
        except (TypeError, ValueError):
            return
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()  # 丢最旧帧，保证新数据优先
                self._queue.put_nowait(payload)
            except (queue.Empty, queue.Full):
                self.dropped += 1

    # ---------- 后台发送线程 ----------

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._socket.sendto(payload, self._addr)
                self.sent += 1
            except OSError:
                # 对端未监听 / 网络错误：遥测允许丢包，静默继续
                pass
