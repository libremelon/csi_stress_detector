import argparse
import asyncio
import csv
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone

try:
    import websockets
    HAS_WEBSOCKETS_LIB = True
except ImportError:
    HAS_WEBSOCKETS_LIB = False

class HRCalculator:
    def __init__(self, rate_size=4):
        self.rate_size = rate_size
        self.rates = [0] * rate_size
        self.rate_spot = 0
        self.beat_avg = 0

    def process(self, hr: int, accuracy: int) -> int:
        if 20 < hr < 255 and accuracy > 0:
            self.rates[self.rate_spot] = int(hr)
            self.rate_spot = (self.rate_spot + 1) % self.rate_size
            self.beat_avg = sum(self.rates) // self.rate_size
        return self.beat_avg


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


def display_hr_reading(hr: int, accuracy: int, calc: HRCalculator):
    beat_avg = calc.process(hr, accuracy)
    if hr <= 20 or hr >= 255 or accuracy == 0:
        sys.stdout.write("Inaccuracte!")
    print(beat_avg)
    sys.stdout.flush()


async def handle_websocket_connection(websocket, logger, calc):
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if logger:
                    logger.log(data)
                hr = data.get("heart_rate", 0)
                accuracy = data.get("accuracy", 0)
                display_hr_reading(hr, accuracy, calc)
            except json.JSONDecodeError:
                pass
    except Exception:
        pass


async def run_websockets_server(host: str, port: int, logger: HRDataLogger, calc: HRCalculator):
    async def handler(*args):
        await handle_websocket_connection(args[0], logger, calc)

    async with websockets.serve(handler, host, port):
        await asyncio.Future()


def run_raw_socket_server(host: str, port: int, logger: HRDataLogger, calc: HRCalculator):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(5)
    try:
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
                                if logger:
                                    logger.log(payload)
                                hr = payload.get("heart_rate", 0)
                                accuracy = payload.get("accuracy", 0)
                                display_hr_reading(hr, accuracy, calc)
                            except json.JSONDecodeError:
                                pass
                except socket.timeout:
                    continue
                except Exception:
                    break
            client.close()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8888)
    parser.add_argument("-o", "--output", type=str, default="watch_hr_log.csv")
    args = parser.parse_args()

    print("Initializing...")

    logger = HRDataLogger(args.output)
    calc = HRCalculator(rate_size=4)

    if HAS_WEBSOCKETS_LIB:
        try:
            asyncio.run(run_websockets_server("0.0.0.0", args.port, logger, calc))
        except KeyboardInterrupt:
            pass
    else:
        run_raw_socket_server("0.0.0.0", args.port, logger, calc)


if __name__ == "__main__":
    main()