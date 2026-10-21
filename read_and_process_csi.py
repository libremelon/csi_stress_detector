#!/usr/bin/env python3
# -*-coding:utf-8-*-

# Copyright 2021 Espressif Systems (Shanghai) PTE LTD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# WARNING: we don't check for Python build-time dependencies until
# check_environment() function below. If possible, avoid importing
# any external libraries here - put in external script, or import in
# their specific function instead.

###
# Modified by Nick Bild
# September 2025
# * Eliminate the graphical interface and calculate amplitude values for each carrier.
# * Add data preprocessing steps from Pulse-Fi paper.
# * Predict heart rate using LSTM model.
###

import os
import time
import sys
import csv
import json
import argparse
import numpy as np
import serial
from io import StringIO
import ast
from scipy.signal import butter, filtfilt, savgol_filter
from typing import List, Tuple, Optional, Dict, Any
import tensorflow as tf
from tensorflow import keras
import numpy as np



# Reduce displayed waveforms to avoid display freezes
CSI_VAID_SUBCARRIER_INTERVAL = 1
csi_vaid_subcarrier_len =0

CSI_DATA_INDEX = 200  # buffer size
CSI_DATA_COLUMNS = 490
DATA_COLUMNS_NAMES_C5C6 = ["type", "id", "mac", "rssi", "rate","noise_floor","fft_gain","agc_gain", "channel", "local_timestamp",  "sig_len", "rx_state", "len", "first_word", "data"]
DATA_COLUMNS_NAMES = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding",
                      "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"]

csi_data_array = np.zeros(
    [CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.float64)
csi_data_phase = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.float64)
csi_data_complex = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.complex64)
agc_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)
fft_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)
fft_gains = []
agc_gains = []




def count_nonempty_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def get_latest_bpm(csv_path="watch_hr_log.csv") -> float:
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in reversed(f.readlines()):
                line = line.strip()
                if line and not line.startswith("Timestamp_ISO"):
                    parts = line.split(",")
                    if len(parts) >= 3:
                        return float(parts[2])
    except Exception:
        pass
    return 0.0

def parse_csi_amplitudes(csi_str):
    # Convert the string into a Python list of ints
    values = ast.literal_eval(csi_str)

    # Reshape into (imag, real) pairs
    complex_pairs = np.array(values).reshape(-1, 2)

    # Convert to complex numbers: real + j*imag
    csi_complex = complex_pairs[:,1] + 1j * complex_pairs[:,0]

    # Compute amplitudes
    amplitudes = np.abs(csi_complex)
    return amplitudes


def remove_dc(signal, fs, lowcut=2.0, highcut=5.0, order=3):
    """
    Remove DC and out-of-band noise using a 3rd-order Butterworth band-pass filter.
    Matches the method described in WiHear (2-5 Hz speaking band).

    Parameters
    ----------
    signal : array-like
        Input CSI amplitude or raw time-series signal.
    fs : float
        Sampling frequency (Hz).
    lowcut : float
        Low cutoff frequency (Hz), default = 2.0 Hz.
    highcut : float
        High cutoff frequency (Hz), default = 5.0 Hz.
    order : int
        Order of Butterworth filter, default = 3.
    
    Returns
    -------
    filtered : ndarray
        Band-pass filtered signal.
    """
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    filtered = filtfilt(b, a, signal)
    return filtered


def butter_bandpass_filter(signal: np.ndarray,
                           lowcut: float,
                           highcut: float,
                           fs: float,
                           order: int = 3) -> np.ndarray:
    """
    Pulse Extraction: 3rd-order Butterworth bandpass (default order=3).
    Uses zero-phase filtering (filtfilt).
    lowcut/highcut in Hz. fs is sampling frequency (Hz).
    """
    x = np.asarray(signal, dtype=float).copy()
    if x.size == 0:
        return x
    nyq = 0.5 * fs
    if not (0 < lowcut < highcut < nyq):
        raise ValueError(f"Invalid bandpass: low={lowcut}, high={highcut}, Nyquist={nyq}")
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    # filtfilt for zero-phase
    return filtfilt(b, a, x)


def savitzky_golay_smooth(signal: np.ndarray,
                          window_length: int = 15,
                          polyorder: int = 3) -> np.ndarray:
    """
    Pulse Shaping: Savitzky-Golay smoothing (preserve waveform shape).
    Ensures window_length is odd and less than signal length.
    If signal too short, returns original signal.
    """
    x = np.asarray(signal, dtype=float).copy()
    n = x.size
    if n == 0:
        return x
    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    if wl < (polyorder + 2):
        raise ValueError("window_length too small for polyorder")
    if wl >= n:
        # fallback: choose the largest odd window smaller than n
        wl_candidate = n - 1
        if wl_candidate % 2 == 0:
            wl_candidate -= 1
        if wl_candidate < (polyorder + 2):
            # can't apply SG; return original
            return x
        wl = wl_candidate
    return savgol_filter(x, wl, polyorder)


def csi_data_read_parse(port: str, csv_writer, log_file_fd,callback=None):
    global fft_gains, agc_gains
    ser = serial.Serial(port=port, baudrate=921600,bytesize=8, parity='N', stopbits=1)
    count =0
    if ser.isOpen():
        print("open success")
        ser.reset_input_buffer()
        time.sleep(0.2)
        ser.readline()  # discard possible partial line
    else:
        print("open failed")
        return
    
    frame_num = 0
    saved_samples = 0
    shaped_data = []
    try:
        while True:

            if os.path.exists("stop.collection"):
                print("stop.collection detected. Halting data collection.")
                try:
                    os.remove("stop.collection")
                except OSError:
                    pass
                break

            if MAX_SAMPLES != -1:
                current_count = saved_samples if COLLECT_TRAINING_DATA else frame_num
                if current_count >= MAX_SAMPLES:
                    print(f"\nReached sample limit ({MAX_SAMPLES} samples). Stopping collection.")
                    break

            frame_num += 1

            raw_bytes = ser.readline()
            if not raw_bytes:
                break
            try:
                line_str = raw_bytes.decode('utf-8', errors='ignore').strip()
            except Exception:
                continue

            index = line_str.find('CSI_DATA')
            if index == -1:
                log_file_fd.write(line_str + '\n')
                log_file_fd.flush()
                continue

            csv_reader = csv.reader(StringIO(line_str))
            try:
                csi_data = next(csv_reader)
            except Exception:
                continue

            if len(csi_data) != len(DATA_COLUMNS_NAMES) and len(csi_data) != len(DATA_COLUMNS_NAMES_C5C6):
                # print("element number is not equal",len(csi_data),len(DATA_COLUMNS_NAMES) )
                log_file_fd.write("element number is not equal\n")
                log_file_fd.write(line_str + '\n')
                log_file_fd.flush()
                continue

            try:
                csi_raw_data = json.loads(csi_data[-1])
            except json.JSONDecodeError:
                log_file_fd.write("data is incomplete\n")
                log_file_fd.write(line_str + '\n')
                log_file_fd.flush()
                continue

            # A valid CSI line must carry a JSON *array* in its last field.
            # json.loads() will also return ints/floats/strings/dicts for
            # malformed lines, which caused the TypeError on len().
            if not isinstance(csi_raw_data, list):
                log_file_fd.write(
                    "csi data payload is not a list (got "
                    f"{type(csi_raw_data).__name__}: {csi_raw_data!r})\n"
                )
                log_file_fd.write(line_str + '\n')
                log_file_fd.flush()
                continue

            # Reject odd-length or empty arrays before DSP.
            if len(csi_raw_data) == 0 or len(csi_raw_data) % 2 != 0:
                log_file_fd.write(
                    f"odd/empty csi payload (len={len(csi_raw_data)})\n"
                )
                log_file_fd.write(line_str + '\n')
                log_file_fd.flush()
                continue

            try:
                csi_data_len = int(csi_data[-3])
            except (ValueError, IndexError):
                continue

            if csi_data_len != len(csi_raw_data):
                # print("csi_data_len is not equal",csi_data_len,len(csi_raw_data))
                log_file_fd.write("csi_data_len is not equal\n")
                log_file_fd.write(line_str + '\n')
                log_file_fd.flush()
                continue

            fft_gain = int(csi_data[6])
            agc_gain = int(csi_data[7])
            
            fft_gains.append(fft_gain)
            agc_gains.append(agc_gain)

            csv_writer.writerow(csi_data)

            
            ###
            # Pulse-Fi CSI data processing steps.
            ###

            # Step 1: Amplitude conversion.
            amplitudes = parse_csi_amplitudes(csi_data[-1])
            # Step 2: Stationary Noise Removal.
            dc_removed = remove_dc(amplitudes, fs=20.0)
            # Step 3: Pulse extraction.
            bandpassed = butter_bandpass_filter(dc_removed, 0.8, 2.17, 20.0, order=3)
            # Step 4: Pulse shaping.
            shaped = savitzky_golay_smooth(bandpassed, window_length=15, polyorder=3)

            if COLLECT_TRAINING_DATA:
                current_bpm = get_latest_bpm("watch_hr_log.csv")
                train.write(','.join(map(str, shaped)) + '\n')
                hr_file.write(f"{current_bpm:.2f}\n")
                train.flush()
                hr_file.flush()
                saved_samples += 1
                if saved_samples % 10 == 0 or saved_samples == 1:
                    print(f"\r[Collected {saved_samples} samples | Current Watch HR: {current_bpm:.1f} BPM]", end="", flush=True)
            else:
                shaped_data.append(shaped)
                if len(shaped_data) > 100:
                    # At this point, shaped_data contains the latest 100 'shaped' arrays.
                    # These are ready to be fed into the LSTM.
                    shaped_data = shaped_data[1:]
                    # Reshape the data for input to the model.
                    shaped_data_np = np.array(shaped_data, dtype=np.float32)
                    shaped_data_np = shaped_data_np.reshape((1, 100, 192))  # shape: (1, 100, 192)
                    # Make a heart rate prediction.
                    new_prediction = model.predict(shaped_data_np, verbose=0)
                    print(new_prediction[0][0])
                    sys.stdout.flush()
                else:
                    print("Collected {0} of 100 initial CSI samples.".format(len(shaped_data)))
                    sys.stdout.flush()


            # Rotate data to the left
            # csi_data_array[:-1] = csi_data_array[1:]
            # csi_data_phase[:-1] = csi_data_phase[1:]
            csi_data_complex[:-1] = csi_data_complex[1:]
            agc_gain_data[:-1] = agc_gain_data[1:]
            fft_gain_data[:-1] = fft_gain_data[1:]
            agc_gain_data[-1] = agc_gain
            fft_gain_data[-1] = fft_gain
            
            for i in range(csi_data_len // 2):
                csi_data_complex[-1][i] = complex(csi_raw_data[i * 2 + 1],
                                                csi_raw_data[i * 2])
    except KeyboardInterrupt:
        print("\n\n[INFO] Stopped by user (Ctrl+C). Saving and closing...")
    finally:
        ser.close()
        try:
            log_file_fd.close()
        except Exception:
            pass
        if COLLECT_TRAINING_DATA:
            train.close()
            hr_file.close()
            try:
                n_csi = count_nonempty_lines("training_data.txt")
                n_hr = count_nonempty_lines("hr_data.txt")
                print(f"[INFO] Files saved cleanly: {n_csi} CSI rows, {n_hr} HR rows in dataset.")
            except Exception:
                pass
        print("[INFO] Serial connection closed.")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read CSI data from serial port and display it graphically")
    parser.add_argument('-p', '--port', dest='port', action='store', required=True,
                        help="Serial port number of csv_recv device")
    parser.add_argument('-s', '--store', dest='store_file', action='store', default='./csi_data.csv',
                        help="Save the data printed by the serial port to a file")
    parser.add_argument('-l', '--log', dest="log_file", action="store", default="./csi_data_log.txt",
                        help="Save other serial data the bad CSI data to a log file")

    parser.add_argument('--collect-training-data', action='store_true', default=False)
    parser.add_argument('--training-append', action='store_true', default=False)
    parser.add_argument('--max-samples', type=int, default=-1)

    args = parser.parse_args()
    COLLECT_TRAINING_DATA = args.collect_training_data
    TRAINING_APPEND = args.training_append
    MAX_SAMPLES = args.max_samples

    if COLLECT_TRAINING_DATA:
        if TRAINING_APPEND:
            n_csi = count_nonempty_lines("training_data.txt")
            n_hr = count_nonempty_lines("hr_data.txt")
            if n_csi != n_hr:
                raise RuntimeError(f"Training files are misaligned: {n_csi} CSI rows vs {n_hr} HR rows")
            print(f"Appending to existing training data ({n_csi} prior rows detected).")
        mode = "a" if TRAINING_APPEND else "w"
        train = open("training_data.txt", mode)
        hr_file = open("hr_data.txt", mode)
    else:
        model = keras.models.load_model("csi_hr.keras", safe_mode=False)
    serial_port = args.port
    file_name = args.store_file
    log_file_name = args.log_file

    try:
        csi_data_read_parse(serial_port, csv.writer(open(file_name, 'w')), open(log_file_name, 'w'))
    except KeyboardInterrupt:
        pass
