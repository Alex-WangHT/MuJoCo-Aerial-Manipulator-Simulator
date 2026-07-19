"""遥测接收端示例：独立进程，监听 UDP 并打印 MujocoSimulation 发来的 JSON 遥测。

用法（先启动接收端，再启动仿真）::

    .venv/Scripts/python tools/telemetry_receiver.py --port 9100
    .venv/Scripts/python main.py --telemetry-out 127.0.0.1:9100

数据报格式：每包一行 JSON，例如（有机械臂时多 joints/ee 两个字段）::

    {"t":1.23,"pos":[0,0,1.5],"vel":[0,0,0],"rpy":[0,0,0],
     "omega":[0,0,0],"thrusts":[3.17,...],"joints":[0,0],"ee":[0,0,1.16]}
"""

from __future__ import annotations

import argparse
import json
import socket


def main() -> None:
    parser = argparse.ArgumentParser(description="MujocoSimulation UDP 遥测接收端示例")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=9100, help="监听端口（默认 9100）")
    parser.add_argument("--max", type=int, default=0,
                        help="收满 N 包后退出（默认 0 = 一直接收）")
    parser.add_argument("--raw", action="store_true",
                        help="打印原始 JSON（默认打印摘要）")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    print(f"[Receiver] 监听 udp://{args.host}:{args.port}，等待遥测…（Ctrl+C 退出）")

    received = 0
    try:
        while True:
            data, addr = sock.recvfrom(65535)
            for line in data.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                packet = json.loads(line)
                received += 1
                if args.raw:
                    print(line)
                else:
                    pos = [round(v, 3) for v in packet["pos"]]
                    rpy = [round(v, 3) for v in packet["rpy"]]
                    extra = ""
                    if "ee" in packet:
                        extra = f", ee={[round(v, 3) for v in packet['ee']]}"
                    print(f"[Receiver] #{received} t={packet['t']:.3f}s pos={pos} rpy={rpy}{extra}")
                if args.max and received >= args.max:
                    print(f"[Receiver] 已收满 {received} 包，退出。")
                    return
    except KeyboardInterrupt:
        print(f"\n[Receiver] 共接收 {received} 包。")


if __name__ == "__main__":
    main()
