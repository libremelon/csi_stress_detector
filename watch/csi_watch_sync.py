#!/usr/bin/env python3
"""
CSI & Galaxy Watch 4 HR Synchronizer
-------------------------------------
Hooks into read_and_process_csi.py to combine ESP32 Wi-Fi Channel State Information (CSI)
with real-time timestamped Heart Rate (HR) ground-truth from Samsung Galaxy Watch 4 Classic.
"""

import os
import sys
import csv
import json
import time
import socket
import threading
import argparse

# Try importing websockets
try:
    import websockets
    import asyncio
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


class WatchHRTracker:
    """Thread-safe store for latest watch heart rate telemetry."""
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_hr = 0
        self.latest_accuracy = 0
        self.latest_timestamp_iso = ""
        self.latest_epoch_ms = 0
        self.received_count = 0

    def update(self, payload: dict):
        with self.lock:
            self.latest_hr = payload.get("heart_rate", 0)
            self.latest_accuracy = payload.get("accuracy", 0)
            self.latest_timestamp_iso = payload.get("timestamp_iso", "")
            self.latest_epoch_ms = payload.get("epoch_ms", int(time.time() * 1000))
            self.received_count += 1

    def get_snapshot(self):
        with self.lock:
            return {
                "hr": self.latest_hr,
                "accuracy": self.latest_accuracy,
                "timestamp_iso": self.latest_timestamp_iso,
                "epoch_ms": self.latest_epoch_ms,
                "count": self.received_count
            }


# Global tracker instance
WATCH_TRACKER = WatchHRTracker()


def start_watch_server_thread(host: str, port: int):
    """Runs the WebSocket server in a background thread."""
    def _run_server():
        if HAS_WEBSOCKETS:
            async def _handler(websocket, path):
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        WATCH_TRACKER.update(data)
                    except json.JSONDecodeError:
                        pass

            async def _main_ws():
                async with websockets.serve(_handler, host, port):
                    await asyncio.Future()

            loop = asyncio.new_event_policy().new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_main_ws())
        else:
            # Native socket fallback
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen(5)
            while True:
                client, _ = server.accept()
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
                                    WATCH_TRACKER.update(payload)
                                except json.JSONDecodeError:
                                    pass
                    except Exception:
                        break
                client.close()

    t = threading.Thread(target=_run_server, daemon=True)
    t.start()
    return t


def patch_and_run_csi(serial_port: str, csi_store_file: str, log_file_name: str, watch_port: int):
    """Integrates Watch HR ground truth into CSI processing."""

    # Import read_and_process_csi module
    import read_and_process_csi as csi_mod

    # Start background watch receiver
    local_ip = csi_mod.socket.gethostbyname(csi_mod.socket.gethostname()) if hasattr(csi_mod.socket, "gethostname") else "0.0.0.0"
    print("=" * 70)
    print("  CSI & GALAXY WATCH 4 CLASSIC HR SYNCHRONIZER  ")
    print("=" * 70)
    print(f"  Watch WebSocket Server : ws://{local_ip}:{watch_port}")
    print(f"  ESP32 CSI Serial Port  : {serial_port}")
    print("=" * 70)

    start_watch_server_thread("0.0.0.0", watch_port)

    # Prepare ground-truth paired CSV file
    gt_file_path = "csi_watch_paired_dataset.csv"
    gt_file_exists = os.path.exists(gt_file_path)
    gt_file = open(gt_file_path, 'a', newline='', encoding='utf-8')
    gt_writer = csv.writer(gt_file)
    if not gt_file_exists:
        gt_writer.writerow(["System_Timestamp", "Watch_Timestamp", "Watch_Epoch_MS", "Watch_HR_GroundTruth", "Watch_Accuracy", "CSI_Shaped_Data"])
        gt_file.flush()

    # Original function wrapper
    def sync_callback(shaped_feature, csi_lstm_predicted_hr=None):
        snapshot = WATCH_TRACKER.get_snapshot()
        sys_ts = csi_mod.datetime.now(csi_mod.timezone.utc).isoformat() if hasattr(csi_mod, "timezone") else str(time.time())
        
        # Write paired dataset row
        gt_writer.writerow([
            sys_ts,
            snapshot["timestamp_iso"],
            snapshot["epoch_ms"],
            snapshot["hr"],
            snapshot["accuracy"],
            json.dumps(shaped_feature.tolist() if hasattr(shaped_feature, "tolist") else list(shaped_feature))
        ])
        gt_file.flush()

        # Display comparison in terminal
        watch_hr_str = f"{snapshot['hr']} BPM" if snapshot["hr"] > 0 else "Waiting for Watch..."
        if csi_lstm_predicted_hr is not None:
            diff = abs(csi_lstm_predicted_hr - snapshot["hr"]) if snapshot["hr"] > 0 else 0.0
            print(f"[SYNC] CSI Predicted: \033[1;33m{csi_lstm_predicted_hr:5.1f} BPM\033[0m | Watch Actual: \033[1;32m{watch_hr_str:<20}\033[0m | Diff: {diff:4.1f} BPM")
        else:
            print(f"[SYNC] CSI Feature Collected | Watch Ground-Truth HR: \033[1;32m{watch_hr_str}\033[0m")

    print("[*] Launching CSI parser with Watch synchronization...")
    # Wrap execution of csi_data_read_parse
    log_fd = open(log_file_name, 'w')
    csv_out = csv.writer(open(csi_store_file, 'w'))

    try:
        csi_mod.csi_data_read_parse(serial_port, csv_out, log_fd, callback=sync_callback)
    except KeyboardInterrupt:
        print("\n[*] Synchronizer stopped.")
    finally:
        gt_file.close()


def main():
    parser = argparse.ArgumentParser(description="Synchronize ESP32 CSI with Galaxy Watch 4 HR")
    parser.add_argument('-p', '--port', dest='port', action='store', required=True, help="Serial port of ESP32 CSI device (e.g. /dev/ttyUSB0)")
    parser.add_argument('-w', '--watch-port', dest='watch_port', action='store', type=int, default=8765, help="WebSocket port for Galaxy Watch 4 (default: 8765)")
    parser.add_argument('-s', '--store', dest='store_file', action='store', default='./csi_data.csv', help="Save raw CSI output CSV")
    parser.add_argument('-l', '--log', dest="log_file", action="store", default="./csi_data_log.txt", help="CSI error log file")
    
    args = parser.parse_args()
    patch_and_run_csi(args.port, args.store_file, args.log_file, args.watch_port)


if __name__ == "__main__":
    main()
