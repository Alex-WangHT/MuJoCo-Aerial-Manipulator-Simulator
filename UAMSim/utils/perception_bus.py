"""传感数据总线：帧同步快照（进程内）+ 感知传输（跨进程 UDP 分片）。

本模块统一承载全部传感数据通路，按实时性分两层：

- **同步层**（控制环，1 kHz）：:class:`SensorSnapshot` 通用快照，仿真线程
  每帧把真值（qpos/qvel）、MJCF 声明的传感器读数与计算量（欧拉角、雅可比）
  打包，经帧同步邮箱阻塞式交给控制器线程——小、快、严格同步；
- **异步层**（感知，各自频率）：相机/雷达等大数据量通道，由
  :class:`PerceptionSource` / :class:`PerceptionPublisher` /
  :class:`PerceptionReceiver` 组成，UDP 分片、降频、允许丢帧，
  绝不阻塞物理主循环。

异步层协议（每个 UDP 数据报，小端）::

    MAGIC 'PB' | u8 version | u32 frame_id | u16 chunk_idx | u16 chunk_cnt
    | f64 timestamp | u16 meta_len | meta(JSON, 含 "name") | payload 分片

单帧 payload 可超过 UDP 单报上限（图像 ~MB 级），按 ``CHUNK_PAYLOAD``
切片发送；接收端按 frame_id 重组，超时未齐的半帧作废。
"""

from __future__ import annotations

import json
import queue
import socket
import struct
import threading
import time

MAGIC = b"PB"
VERSION = 1
CHUNK_PAYLOAD = 60000          # 单片 payload 上限（数据报 < 64 KiB）
_HEADER = struct.Struct("<2sBIHHdH")   # magic, ver, frame_id, idx, cnt, t, meta_len


class SensorSnapshot(dict):
    """通用传感快照（帧同步邮箱的载荷类型）。

    命名字段字典 + 属性访问（``snap["dronePosition"]`` 与
    ``snap.dronePosition`` 等价），字段由发布方按机器人构型填充——
    不再为每种机器人手写死 dataclass，新增传感器/计算量只需加一个键。

    约定字段：
    - 公共：``timestamp`` / ``timestep``
    - 多旋翼：``dronePosition`` / ``droneVelocity`` / ``droneOrientation``
      （欧拉角）/ ``droneAngularVelocity``
    - 机械臂：``jointPositions`` / ``jointVelocities`` / ``eePosition`` /
      ``eeJacobian``
    快照全部为当帧拷贝，订阅线程只读本快照、不触碰 mjData。
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value) -> None:
        self[key] = value


class PerceptionSource:
    """感知源基类（预留接口）：相机 / 雷达落地时继承本类实现 ``capture``。

    - ``name``：通道名（如 ``"camera_front"`` / ``"lidar"``），对端按名订阅；
    - ``rate_hz``：期望采样频率，仿真线程按物理步长折算成抽稀因子；
    - ``capture(model, data) -> (payload, meta)``：**只能由仿真线程调用**，
      返回一帧二进制数据（如 JPEG 字节流 / 点云 ``ndarray.tobytes()``）与
      元信息字典（如 ``{"shape": [480, 640, 3], "dtype": "uint8"}``）。
    """

    name: str = ""
    rate_hz: float = 10.0

    def capture(self, model, data) -> tuple[bytes, dict]:
        raise NotImplementedError


class PerceptionPublisher:
    """把感知帧经 UDP 发往另一个进程（分片、非阻塞、独立线程）。

    参数:
        host: 目标主机（本机调试用 ``127.0.0.1``）
        port: 目标 UDP 端口
        max_queue: 待发送分片队列上限（满时丢最旧片，保护主循环）
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9200, max_queue: int = 512):
        self._addr = (host, int(port))
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=max_queue)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="PerceptionPublisher"
        )
        self._frame_id = 0
        self.sent_frames = 0
        self.dropped_chunks = 0

    # ---------- 生命周期 ----------

    def start(self) -> None:
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        try:
            self._socket.close()
        except OSError:
            pass

    # ---------- 主循环侧（仿真线程调用） ----------

    def publish(self, name: str, payload: bytes, meta: dict | None = None) -> None:
        """切片入队一帧感知数据，立即返回。**永不阻塞、永不抛异常**。"""
        try:
            meta_bytes = json.dumps(
                {"name": name, **(meta or {})}, separators=(",", ":")
            ).encode("utf-8")
            if len(meta_bytes) > 65535:
                return
            payload = bytes(payload)
            self._frame_id += 1
            frame_id = self._frame_id
            timestamp = time.time()
            n_chunks = max(1, (len(payload) + CHUNK_PAYLOAD - 1) // CHUNK_PAYLOAD)
            for idx in range(n_chunks):
                datagram = _HEADER.pack(
                    MAGIC, VERSION, frame_id, idx, n_chunks, timestamp, len(meta_bytes)
                ) + meta_bytes + payload[idx * CHUNK_PAYLOAD:(idx + 1) * CHUNK_PAYLOAD]
                try:
                    self._queue.put_nowait(datagram)
                except queue.Full:
                    try:
                        self._queue.get_nowait()      # 丢最旧片，保证新数据优先
                        self._queue.put_nowait(datagram)
                    except (queue.Empty, queue.Full):
                        self.dropped_chunks += 1
            self.sent_frames += 1
        except (TypeError, ValueError, struct.error):
            return

    # ---------- 后台发送线程 ----------

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                datagram = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._socket.sendto(datagram, self._addr)
            except OSError:
                # 对端未监听 / 网络错误：感知允许丢帧，静默继续
                pass


class PerceptionReceiver:
    """对端进程的接收/重组器：后台线程收片，完整帧入队供消费。

    用法::

        receiver = PerceptionReceiver(port=9200)
        receiver.start()
        meta, payload = receiver.recv(timeout=1.0)   # None 表示超时
        receiver.stop()

    乱序、丢片、半帧（超时未齐）自动丢弃；完整帧队列满时丢最旧帧。
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9200,
                 max_frames: int = 8, frame_timeout: float = 0.5):
        self._addr = (host, int(port))
        self._max_frames = max_frames
        self._frame_timeout = frame_timeout
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._frames: queue.Queue[tuple[dict, bytes]] = queue.Queue(maxsize=max_frames)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="PerceptionReceiver"
        )
        self.received_frames = 0
        self.expired_frames = 0

    # ---------- 生命周期 ----------

    def start(self) -> None:
        # 大帧突发（图像每帧数十片）需要足够大的接收缓冲，否则 OS 直接丢报
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self._socket.bind(self._addr)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        try:
            self._socket.close()
        except OSError:
            pass

    # ---------- 消费侧 ----------

    def recv(self, timeout: float | None = None) -> tuple[dict, bytes] | None:
        """取一帧完整数据，返回 ``(meta, payload)``；超时返回 None。"""
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    # ---------- 后台接收线程 ----------

    def _worker(self) -> None:
        self._socket.settimeout(0.2)
        partials: dict[int, dict] = {}   # frame_id -> 重组上下文
        while not self._stop_event.is_set():
            try:
                datagram, _ = self._socket.recvfrom(65535)
            except (socket.timeout, OSError):
                self._expire(partials)
                continue
            try:
                magic, ver, frame_id, idx, cnt, timestamp, meta_len = _HEADER.unpack_from(datagram)
            except struct.error:
                continue
            if magic != MAGIC or ver != VERSION or cnt == 0 or idx >= cnt:
                continue
            meta_end = _HEADER.size + meta_len
            try:
                meta = json.loads(datagram[_HEADER.size:meta_end].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

            ctx = partials.setdefault(frame_id, {
                "meta": meta, "cnt": cnt, "t": timestamp,
                "chunks": {}, "deadline": time.monotonic() + self._frame_timeout,
            })
            ctx["chunks"][idx] = datagram[meta_end:]
            if len(ctx["chunks"]) == ctx["cnt"]:
                del partials[frame_id]
                payload = b"".join(ctx["chunks"][i] for i in range(ctx["cnt"]))
                try:
                    self._frames.put_nowait((ctx["meta"], payload))
                    self.received_frames += 1
                except queue.Full:
                    try:
                        self._frames.get_nowait()     # 丢最旧帧
                        self._frames.put_nowait((ctx["meta"], payload))
                        self.received_frames += 1
                    except (queue.Empty, queue.Full):
                        pass
            self._expire(partials)

    def _expire(self, partials: dict) -> None:
        now = time.monotonic()
        stale = [fid for fid, ctx in partials.items() if now > ctx["deadline"]]
        for fid in stale:
            del partials[fid]
            self.expired_frames += 1
