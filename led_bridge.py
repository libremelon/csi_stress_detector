#!/usr/bin/env python3
"""Reads hr_dashboard/heart_rate_output.json, computes the stress index exactly like
hr_dashboard/index.html, and sends 'S:<stress>' or 'OFF' to the ESP32 once per second."""
import argparse
import json
import math
import subprocess
import time

REST_BPM = 60.0
MAX_BPM = 120.0
MIN_READINGS = 10
MAX_POINTS = 120
MEAN_WINDOW = 30
STALE_SECONDS = 10.0
SEND_INTERVAL = 1.0


def read_reading(path):
    """Return (timestamp, heart_rate) or None if there is no valid reading."""
    try:
        with open(path, "r") as f:
            text = f.read().strip()
        try:
            obj = json.loads(text)
        except ValueError:
            lines = [l for l in text.splitlines() if l.strip()]
            obj = json.loads(lines[-1])
        if isinstance(obj, list):
            obj = obj[-1]
        hr = float(obj["heart_rate_bpm"])
        ts = float(obj["timestamp"])
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None
    if not (math.isfinite(hr) and math.isfinite(ts)) or hr <= 0:
        return None
    return ts, hr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", help="serial port of the ESP32, e.g. /dev/ttyUSB0")
    ap.add_argument("-b", "--baud", default="115200")
    ap.add_argument("-f", "--file", default="hr_dashboard/heart_rate_output.json")
    ap.add_argument("--dry-run", action="store_true", help="print only, do not open the serial port")
    args = ap.parse_args()

    ser = None
    if not args.dry_run:
        if not args.port:
            ap.error("--port is required unless --dry-run is used")
        subprocess.run(["stty", "-F", args.port, args.baud, "raw", "-echo", "-hupcl"], check=True)
        ser = open(args.port, "wb", buffering=0)

    readings = []
    last_ts = None
    last_change = 0.0
    try:
        while True:
            r = read_reading(args.file)
            now = time.monotonic()
            if r is not None and r[0] != last_ts:
                last_ts = r[0]
                last_change = now
                readings.append(r[1])
                if len(readings) > MAX_POINTS:
                    readings.pop(0)

            fresh = last_ts is not None and (now - last_change) <= STALE_SECONDS
            if fresh and len(readings) >= MIN_READINGS:
                window = readings[-MEAN_WINDOW:]
                stress = (sum(window) / len(window) - REST_BPM) / (MAX_BPM - REST_BPM) * 100.0
                stress = max(0.0, min(100.0, stress))
                line = "S:%.1f\n" % stress
            else:
                line = "OFF\n"

            if ser:
                ser.write(line.encode())
            print(line.strip(), flush=True)
            time.sleep(SEND_INTERVAL)
    except KeyboardInterrupt:
        pass
    finally:
        if ser:
            try:
                ser.write(b"OFF\n")
            except OSError:
                pass
            ser.close()


if __name__ == "__main__":
    main()
