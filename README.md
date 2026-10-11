# CSI Stress/Lie detector

This project uses Wi-Fi CSI, a non invasive, low cost way to estimate heart rate based on the principles outlined by the PulseFi research paper. This repository is hardforked from [nickbild/csi_hr](https://github.com/nickbild/csi_hr) and would've been very difficult for me to otherwise implement from scratch.

This project differs from the original in the sense:

1. I've swapped the MAX30102 sensor with a Galaxy Watch Series 4. As a result the arduino code is removed. The Samsung watch relays the realtime heart rate data over a particular port which I then use inside `watch_hr.py`. The Samsung watch app is not opensourced (yet, because I didn't make it)
2. The hardcoded constants in `read_and_process_csi.py` are replaced with command line arguments.

- Added an option to append the training data instead of starting over every time (So you can work, pause the program, go out for a jog, and resume data collection to collect heart rate data over a wider range of scenarios).
- Added an option to stop the data collection gracefully (create a stop file with `touch stop.collection` to stop data collection)
- New command line arguments for `read_and_process_csi.py`

1.  `--collect-training-data` instructs the program to collect csi and hr data which will later be used for training.
2.  `--training-append`. If this flag is not passed then the older training data is overwritten.
3.  `--max-samples`. Specifies the maximum number of samples that the program should collect during data the data collection phase.
4.  `master_menu.sh`: Makes it simpler to run the project, without you have to remember lots of commands. Execute it and follow the steps in order.

- Do note that I hardcoded the watch's private IPv4 address in $case 1 of the script so update it. The IPv4 address can be found under wireless debugging under developer options on the Galaxy Watch 4 (It only needs to be updated once!).

2. `hr_data.txt` is now created under `read_and_process_csi.py` which wasn't being done earlier, even though it's required by the main training script `train.py`, and handled gracefully (along with `training_data.txt`).
3. Created a `requirements.txt` for easier venv creation. If CUDA isn't being detected, despite drivers being installed then it's probably becaused the `LD_LIBRARY_PATH` isn't being set.

The Stress/Lie detection part will be handled via a separate frontend which my friends will work on and I'll link it when it's done.

## Repository Structure

```
.
├── csi_send/               # ESP32 CSI transmitter firmware copied from https://github.com/espressif/esp-csi
├── csi_recv/               # ESP32 CSI receiver firmware from the same repo
├── watch/
│   └── watch_hr.py         # Samsung Galaxy Watch heart rate data collector
├── master_menu.sh          # Helper script to navigate this project
├── read_and_process_csi.py # Meat of the PulseFi research paper with the heavy lifting done by Mr. Nickbild. Parses CSI, filters it and extracts features
├── train.py                # TensorFlow LSTM model training script
└── requirements.txt        # Python dependencies
```

---

## Hardware & Software Requirements

### Hardware

- **CSI Transmitter:** ESP32 development board (running `csi_send`).
- **CSI Receiver:** ESP32 development board (running `csi_recv`) connected via USB serial.
- **Reference HR Sensor:** Samsung Galaxy Watch running Wear OS.
- **Host System:** Linux machine with CUDA-capable GPU (recommended for training).

### Software

- **ESP-IDF v4.x / v5.x**
- **Python 3.11+**
- **TensorFlow** (with CUDA GPU support)
- Additional Python dependencies listed in `requirements.txt`

---

## Setup & Installation

### 1. Python Environment Setup

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### 2. ESP32 Firmware Flashing

**Transmitter Setup (`csi_send`):**

```bash
cd csi_send
idf.py set-target esp32
idf.py build flash monitor
cd ..
```

**Receiver Setup (`csi_recv`):**

```bash
cd csi_recv
idf.py set-target esp32
idf.py build flash monitor
cd ..
```

---

## Usage Workflow

The repository includes a unified menu system to manage hardware telemetry streams, sample limits, and training routines.

### Master Automation Menu

Launch the control interface:

```bash
chmod +x master_menu.sh
./master_menu.sh
```

---

## Attribution & Credits

- **Original Base Project:** Hardforked from [csi_hr](https://github.com/nickbild/csi_hr) by **Nick Bild**.
- **CSI Components:** Derived from Espressif's [esp-csi](https://github.com/espressif/esp-csi) library.
- **Galaxy Watch Module:** Galaxy watch app by [@GiridharNaidu4152](https://github.com/GiridharNaidu4152).

---

## AI Disclosure

Parts of this README and codebase were developed with AI assistance; however, all code and documentation have been reviewed, verified, and tested by me (no 'vibecoding' per se).
