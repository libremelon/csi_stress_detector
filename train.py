import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import glob
import json
import datetime


MSE_THRESHOLD = 0.01
CONTINUE_TRAINING_MODEL = ""
HR_DATA_PATH = "/content/drive/MyDrive/ecs-data/combined/hr_data.txt"
TRAINING_DATA_PATH = "/content/drive/MyDrive/ecs-data/combined/training_data.txt"
MODEL_SAVE_PATH = "/content/drive/MyDrive/ecs-data/csi_hr.keras"
TRAIN_STOP_FILE_PATH = "/content/drive/MyDrive/ecs-data/training.stop"
TOTAL_EPOCHS = 100                 # total, not additional

CHECKPOINT_DIR = "/content/drive/MyDrive/ecs-data/checkpoints"
LATEST_CKPT = os.path.join(CHECKPOINT_DIR, "csi_hr_latest.keras")
EPOCH_FILE = os.path.join(CHECKPOINT_DIR, "last_epoch.json")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def last_completed_epoch():
    if os.path.isfile(EPOCH_FILE):
        try:
            with open(EPOCH_FILE) as f:
                return json.load(f)["epoch"]
        except (ValueError, KeyError, OSError):
            print("WARNING: could not read {0}; restarting epoch count.".format(EPOCH_FILE))
    return -1


class EpochTracker(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Write after ModelCheckpoint fires (Keras orders callbacks by list order)
        with open(EPOCH_FILE, "w") as f:
            json.dump({"epoch": epoch}, f)


data = []
data_hr = []
train_x = []
train_y = []

print("Reading in training data ({0})...".format(datetime.datetime.now()))

# 1. Read HR data first
raw_hr = []
with open(HR_DATA_PATH, "r") as f:
    for line in f:
        cleaned = line.strip()
        if cleaned:
            raw_hr.append(float(cleaned))

# 2. Read CSI data and strictly validate 192 features per line
EXPECTED_FEATURES = 192
dropped_lines = 0

with open(TRAINING_DATA_PATH, "r") as f:
    for idx, line in enumerate(f):
        cleaned = line.strip()
        if not cleaned:
            continue
        
        pieces = cleaned.split(",")
        # Check feature count and ensure corresponding HR entry exists
        if len(pieces) == EXPECTED_FEATURES and idx < len(raw_hr):
            try:
                row = [float(x) for x in pieces]
                data.append(row)
                data_hr.append(raw_hr[idx])
            except ValueError:
                dropped_lines += 1
        else:
            dropped_lines += 1

print("Done reading in game data ({0}). Dropped {1} invalid/mismatched lines.".format(
    datetime.datetime.now(), dropped_lines
))

initial_epoch = 0

if CONTINUE_TRAINING_MODEL != "":
    print("Loading saved model: {0}".format(CONTINUE_TRAINING_MODEL))
    model = keras.models.load_model(CONTINUE_TRAINING_MODEL)   # compiled state restored
    initial_epoch = last_completed_epoch() + 1
elif os.path.isfile(LATEST_CKPT):
    print("Found checkpoint, auto-resuming from {0}".format(LATEST_CKPT))
    model = keras.models.load_model(LATEST_CKPT)
    initial_epoch = last_completed_epoch() + 1
else:
    # Build the model.
    main_input = keras.Input(shape=(100, 192), name='main_input')
    layers = keras.layers.LSTM(64, return_sequences=True, name='lstm_1')(main_input)
    layers = keras.layers.Dropout(0.2, name='dropout_1')(layers)
    layers = keras.layers.LSTM(32, name='lstm_2')(layers)
    layers = keras.layers.Dropout(0.2, name='dropout_2')(layers)
    layers = keras.layers.Dense(16, activation='relu', name='dense_1')(layers)
    hr_output = keras.layers.Dense(1, name='hr_output')(layers)
    model = keras.Model(inputs=main_input, outputs=hr_output)

    # Compile the model.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss={'hr_output': 'mse'})

print("Resuming at epoch {0}".format(initial_epoch))

# Print the model summary.
model.summary()

# Prepare the training data.
data = np.array(data, dtype=np.float32)
data_hr = np.array(data_hr, dtype=np.float32)

# Chronological split to avoid data leakage: first 80% train, last 20% validation.
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
val_data = data[split_idx:]
train_hr = data_hr[:split_idx]
val_hr = data_hr[split_idx:]

WINDOW_SIZE = 100

# Build windows for the training split only.
train_x = []
train_y = []
for i in range(len(train_data) - WINDOW_SIZE):
    train_x.append(train_data[i : i + WINDOW_SIZE])  # shape: (100, 192)
    # Average heart rate across the 100-frame window
    avg = sum(train_hr[i : i + WINDOW_SIZE]) / WINDOW_SIZE
    train_y.append(avg)

train_x = np.asarray(train_x, dtype=np.float32)
train_y = np.asarray(train_y, dtype=np.float32)

# Build windows for the validation split with an identical loop.
val_x = []
val_y = []
for i in range(len(val_data) - WINDOW_SIZE):
    val_x.append(val_data[i : i + WINDOW_SIZE])  # shape: (100, 192)
    # Average heart rate across the 100-frame window
    avg = sum(val_hr[i : i + WINDOW_SIZE]) / WINDOW_SIZE
    val_y.append(avg)

val_x = np.asarray(val_x, dtype=np.float32)
val_y = np.asarray(val_y, dtype=np.float32)

# Z-score normalization using statistics from the training set only.
train_mean = np.mean(train_x)
train_std = np.std(train_x)
train_x = (train_x - train_mean) / (train_std + 1e-7)
val_x = (val_x - train_mean) / (train_std + 1e-7)

print("Training data X shape: {0}".format(train_x.shape))
print("Training data Y shape: {0}".format(train_y.shape))
print("Validation data X shape: {0}".format(val_x.shape))
print("Validation data Y shape: {0}".format(val_y.shape))

# Train the model.
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
)

callbacks_list = [
    early_stopping,
    keras.callbacks.ModelCheckpoint(
        filepath=LATEST_CKPT,
        save_best_only=True,
        monitor='val_loss',
        save_weights_only=False,   # <-- keep optimizer state
        save_freq='epoch',
        verbose=1,
    ),
    EpochTracker(),
]

if initial_epoch >= TOTAL_EPOCHS:
    print("Checkpoint already at epoch {0} of {1}; nothing left to train.".format(
        initial_epoch, TOTAL_EPOCHS))
else:
    model.fit(
        train_x,
        train_y,
        batch_size=32,
        epochs=TOTAL_EPOCHS,
        initial_epoch=initial_epoch,
        verbose=2,
        validation_data=(val_x, val_y),
        callbacks=callbacks_list,
    )

model.save(MODEL_SAVE_PATH)
print("Model training complete!")