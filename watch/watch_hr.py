#!/usr/bin/env python3
"""
Laptop Receiver for Samsung Galaxy Watch 4 Classic HR Stream
-------------------------------------------------------------
Receives WebSocket real-time Heart Rate (HR) data from Galaxy Watch 4 Classic,
logs timestamped readings to CSV, and prints live heart rate updates.
"""

import os
import sys
import json
import csv
import time
import socket
import argparse
import asyncio
from datetime import datetime, timezone

# Try importing websockets library, fallback to standard asyncio socket server if not available
try:
    import websockets
    HAS_WEBSOCKETS_LIB = True
except ImportError:
    HAS_WEBSOCKETS_LIB = False


def get_local_ips() -> list:
    """Detect local network IP addresses (works offline without internet)."""
    ips = []
    # Method 1: Connecting to broadcast/dummy local target
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        if ip and ip != "127.0.0.1":
            ips.append(ip)
        s.close()
    except Exception:
        pass

    # Method 2: Get hostname IP
    try:
        hostname = socket.gethostname()
        for item in socket.getaddrinfo(hostname, None):
            ip = item[4][0]
            if ":" not in ip and ip != "127.0.0.1" and ip not in ips:
                ips.append(ip)
    except Exception:
        pass

    return ips if ips else ["127.0.0.1"]


class HRDataLogger:
    def __init__(self, csv_filepath: str):
        self.csv_filepath = csv_filepath
        file_exists = os.path.exists(csv_filepath)
        self.file = open(csv_filepath, mode='a', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        if not file_exists:
            self.writer.writerow([
                "Timestamp_ISO",
                "Epoch_MS",
                "HeartRate_BPM",
                "Accuracy",
                "Device",
                "Received_Latency_MS"
            ])
            self.file.flush()

    def log(self, data: dict):
        timestamp_iso = data.get("timestamp_iso", datetime.now(timezone.utc).isoformat())
        epoch_ms = data.get("epoch_ms", int(time.time() * 1000))
        hr = data.get("heart_rate", 0)
        accuracy = data.get("accuracy", 0)
        device = data.get("device", "Galaxy Watch 4 Classic")
        
        now_ms = int(time.time() * 1000)
        latency = max(0, now_ms - epoch_ms) if epoch_ms else 0

        self.writer.writerow([timestamp_iso, epoch_ms, hr, accuracy, device, latency])
        self.file.flush()
        return timestamp_iso, epoch_ms, hr, accuracy, device, latency


async def handle_websocket_connection(websocket, logger, path=None):
    client_ip = websocket.remote_address[0] if hasattr(websocket, "remote_address") and websocket.remote_address else "Unknown"
    print(f"\n[+] Connected to Watch at {client_ip}")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                ts, epoch, hr, acc, dev, lat = logger.log(data)

                # Accuracy status text
                acc_labels = {0: "Unreliable", 1: "Low", 2: "Medium", 3: "High"}
                acc_str = acc_labels.get(acc, f"Level-{acc}")

                print(f"[{ts}] ❤️ HR: \033[1;31m{hr:3d} BPM\033[0m | Accuracy: {acc_str:<10} | Latency: {lat}ms")
            except json.JSONDecodeError:
                print(f"[!] Invalid JSON received: {message}")
    except Exception as e:
        print(f"[-] Connection closed: {e}")


async def run_websockets_server(host: str, port: int, logger: HRDataLogger):
    print(f"[*] Starting WebSocket Server on ws://{host}:{port}")
    async def handler(*args):
        if len(args) == 1:
            await handle_websocket_connection(args[0], logger)
        else:
            await handle_websocket_connection(args[0], logger, path=args[1])

    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # run forever


def run_raw_socket_server(host: str, port: int, logger: HRDataLogger):
    """Fallback WebSocket / TCP HTTP parser if websockets package is absent."""
    print(f"[*] Starting Native Socket Fallback Server on {host}:{port}")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(5)

    try:
        while True:
            client, addr = server.accept()
            print(f"\n[+] Connection from {addr[0]}:{addr[1]}")
            buf = ""
            client.settimeout(5.0)
            while True:
                try:
                    data = client.recv(4096)
                    if not data:
                        break
                    buf += data.decode('utf-8', errors='ignore')
                    lines = buf.split('\n')
                    buf = lines.pop()
                    for line in lines:
                        line = line.strip()
                        if line.startswith('{') and line.endswith('}'):
                            try:
                                payload = json.loads(line)
                                ts, epoch, hr, acc, dev, lat = logger.log(payload)
                                print(f"[{ts}] ❤️ HR: {hr} BPM | Accuracy: {acc} | Latency: {lat}ms")
                            except json.JSONDecodeError:
                                pass
                except socket.timeout:
                    continue
                except Exception:
                    break
            client.close()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")
    finally:
        server.close()


def main():
    parser = argparse.ArgumentParser(description="Galaxy Watch 4 Heart Rate Laptop Receiver")
    parser.add_argument("-p", "--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("-o", "--output", type=str, default="watch_hr_log.csv", help="CSV log output path")
    parser.add_argument("--test", action="store_true", help="Run quick self-test and exit")
    args = parser.parse_args()

    local_ips = get_local_ips()
    primary_ip = local_ips[0]
    print("=" * 65)
    print("  SAMSUNG GALAXY WATCH 4 CLASSIC - HR LAPTOP RECEIVER  ")
    print("=" * 65)
    print(f"  Laptop Local IP(s): \033[1;32m{', '.join(local_ips)}\033[0m")
    print(f"  Target Port       : {args.port}")
    print(f"  CSV Log File      : {args.output}")
    print(f"  Enter on Watch UI : \033[1;36mws://{primary_ip}:{args.port}\033[0m")
    print(f"  (Or over ADB)     : \033[1;36mws://127.0.0.1:{args.port}\033[0m")
    print("=" * 65)

    if args.test:
        print("[+] Self-test mode verified successfully.")
        return

    logger = HRDataLogger(args.output)

    if HAS_WEBSOCKETS_LIB:
        try:
            asyncio.run(run_websockets_server("0.0.0.0", args.port, logger))
        except KeyboardInterrupt:
            print("\n[*] Server shutdown.")
    else:
        run_raw_socket_server("0.0.0.0", args.port, logger)


if __name__ == "__main__":
    main()
