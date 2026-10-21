import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import datetime


MSE_THRESHOLD = 0.01
CONTINUE_TRAINING_MODEL = ""
HR_DATA_PATH = "/content/drive/MyDrive/ecs-data/exerted/hr_data.txt"
TRAINING_DATA_PATH = "/content/drive/MyDrive/ecs-data/exerted/training_data.txt"
MODEL_SAVE_PATH = "/content/drive/MyDrive/ecs-data/csi_hr.keras"
TRAIN_STOP_FILE_PATH = "/content/drive/MyDrive/ecs-data/training.stop"


class stopCallback(keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss <= MSE_THRESHOLD or os.path.isfile(TRAIN_STOP_FILE_PATH):   
            print("\nReached {0} MSE; stopping training.".format(MSE_THRESHOLD)) 
            self.model.stop_training = True


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

if CONTINUE_TRAINING_MODEL != "":
    print("Loading saved model: {0}".format(CONTINUE_TRAINING_MODEL))
    model = keras.models.load_model(CONTINUE_TRAINING_MODEL)
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
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, loss={'hr_output': 'mse'})

# Print the model summary.
model.summary()

# Prepare the training data.
data = np.array(data, dtype=np.float32)

WINDOW_SIZE = 100
for i in range(len(data) - WINDOW_SIZE):
    train_x.append(data[i : i + WINDOW_SIZE])  # shape: (100, 192)
    # Average heart rate across the 100-frame window
    avg = sum(data_hr[i : i + WINDOW_SIZE]) / WINDOW_SIZE
    train_y.append(avg)

train_x = np.asarray(train_x, dtype=np.float32)
train_y = np.asarray(train_y, dtype=np.float32)

print("Training data X shape: {0}".format(train_x.shape))
print("Training data Y shape: {0}".format(train_y.shape))

# Train the model.
callbacks_list = [stopCallback()]

model.fit(
    train_x,
    train_y,
    batch_size=32,
    epochs=5000,
    verbose=2,
    validation_split=0.2,
    callbacks=callbacks_list
)

model.save(MODEL_SAVE_PATH)
print("Model training complete!")