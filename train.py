"""Train an LSTM that predicts heart rate (bpm) from windows of Wi-Fi CSI.

Run in Colab with Drive mounted. Safe to re-run after a disconnect: it resumes
from the last finished epoch, including optimizer state and early-stopping state.

Outputs (all under DATA_DIR):
  csi_hr_v2.keras        final model = best epoch. Takes RAW CSI windows
                         (100, 192) and returns bpm; normalisation is inside it.
  checkpoints_v2/        latest + best checkpoints, resume state, metrics.json
  training.stop          create this file to stop cleanly after the current epoch
"""
import datetime
import hashlib
import json
import math
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from numpy.lib.stride_tricks import sliding_window_view

# ------------------------------- configuration -------------------------------
DATA_DIR = os.environ.get("ECS_DATA_DIR", "/content/drive/MyDrive/ecs-data")
HR_DATA_PATH = os.path.join(DATA_DIR, "combined", "hr_data.txt")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "combined", "training_data.txt")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "csi_hr_v2.keras")   # v2: does not overwrite the old model
TRAIN_STOP_FILE_PATH = os.path.join(DATA_DIR, "training.stop")

CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints_v2")     # new dir: old v1 checkpoints are not resumed
LATEST_CKPT = os.path.join(CHECKPOINT_DIR, "csi_hr_latest.keras")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "csi_hr_best.keras")
STATE_PATH = os.path.join(CHECKPOINT_DIR, "run_state.json")
METRICS_PATH = os.path.join(CHECKPOINT_DIR, "metrics.json")

EXPECTED_FEATURES = 192
WINDOW_SIZE = 100                  # frames per sample; must span several heartbeats (see notes)
HR_MIN, HR_MAX = 30.0, 220.0       # HR readings outside this are sensor dropouts, not data
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # chronological; the remaining 15% is a held-out test set

TOTAL_EPOCHS = int(os.environ.get("TOTAL_EPOCHS", 100))   # total, not additional
BATCH_SIZE = 64
TRAIN_STRIDE = 1                   # raise to 2-4 for faster epochs (neighbouring windows overlap ~99%)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HUBER_DELTA = 3.0                  # bpm; errors beyond this are weighted linearly (robust to HR label noise)
MONITOR = "val_mae"                # model selection / early stopping metric, in bpm
PATIENCE = 15
MIN_DELTA = 1e-3
SEED = 42

CONV_FILTERS, LSTM_1, LSTM_2, DENSE_UNITS = 96, 96, 64, 32
DROPOUT, INPUT_NOISE = 0.3, 0.1

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
keras.utils.set_random_seed(SEED)
PyDataset = getattr(keras.utils, "PyDataset", None) or keras.utils.Sequence

print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))


# ------------------------------- data loading --------------------------------
def parse_csi_line(line):
    pieces = line.split(",")
    if len(pieces) != EXPECTED_FEATURES:
        return None
    try:
        row = np.array(pieces, dtype=np.float32)
    except ValueError:
        return None
    return row if np.isfinite(row).all() else None


def load_aligned_data():
    """Return (csi[N,192], hr[N], valid[N]). Row i of both files describes the same frame.

    Alignment is by position among non-blank lines. Bad rows are KEPT and flagged
    invalid, so time stays aligned and no window ever silently spans a gap.
    """
    hr_vals = []
    with open(HR_DATA_PATH, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    hr_vals.append(float(s))
                except ValueError:
                    hr_vals.append(float("nan"))

    rows = []
    with open(TRAINING_DATA_PATH, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(parse_csi_line(s))

    if len(hr_vals) != len(rows):
        print("WARNING: {0} HR lines vs {1} CSI lines; using the first {2} of each.".format(
            len(hr_vals), len(rows), min(len(hr_vals), len(rows))))
    n = min(len(hr_vals), len(rows))

    csi = np.zeros((n, EXPECTED_FEATURES), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        if rows[i] is not None:
            csi[i] = rows[i]
            valid[i] = True
    hr = np.array(hr_vals[:n], dtype=np.float64)
    hr_ok = np.isfinite(hr) & (hr >= HR_MIN) & (hr <= HR_MAX)
    print("Frames: {0} | bad CSI lines: {1} | bad HR values: {2}".format(
        n, int((~valid).sum()), int((~hr_ok).sum())))
    valid &= hr_ok
    return csi, hr, valid


def window_starts(valid, lo, hi, stride):
    """Start rows of every window that lies fully inside rows [lo, hi) and has no bad frame."""
    n_win = (hi - lo) - WINDOW_SIZE + 1
    if n_win <= 0:
        return np.empty(0, dtype=np.int64)
    bad = np.concatenate([[0], np.cumsum(~valid[lo:hi])])
    ok = (bad[WINDOW_SIZE:WINDOW_SIZE + n_win] - bad[:n_win]) == 0
    return (lo + np.nonzero(ok)[0])[::stride].astype(np.int64)


class WindowSequence(PyDataset):
    """Builds (batch, WINDOW_SIZE, 192) windows on the fly from a sliding-window view,
    so memory stays at one copy of the CSI array instead of ~100x."""

    def __init__(self, csi, targets, starts, shuffle, seed=0):
        try:
            super().__init__(workers=1, use_multiprocessing=False, max_queue_size=8)
        except TypeError:          # older Keras Sequence
            super().__init__()
        self.windows = sliding_window_view(csi, WINDOW_SIZE, axis=0)   # (N-W+1, F, W), no copy
        self.targets = targets
        self.starts = np.asarray(starts)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.starts))
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return math.ceil(len(self.starts) / BATCH_SIZE)

    def __getitem__(self, idx):
        s = self.starts[self.order[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]]
        x = np.ascontiguousarray(self.windows[s].transpose(0, 2, 1))
        return x, self.targets[s].astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)

    def y_true(self):
        return self.targets[self.starts]


# ------------------------------ prepare the data ------------------------------
print("Reading in training data ({0})...".format(datetime.datetime.now()))
csi, hr, valid = load_aligned_data()
N = len(csi)

# Target = mean HR over the window (same definition as before), via cumulative sums.
cum = np.concatenate([[0.0], np.cumsum(np.where(valid, hr, 0.0))])
window_mean_hr = (cum[WINDOW_SIZE:] - cum[:-WINDOW_SIZE]) / WINDOW_SIZE     # index i -> hr[i:i+W]

# Chronological train / val / test. Windows never cross a split boundary.
a, b = int(N * TRAIN_FRAC), int(N * (TRAIN_FRAC + VAL_FRAC))
train_starts = window_starts(valid, 0, a, TRAIN_STRIDE)
val_starts = window_starts(valid, a, b, 1)
test_starts = window_starts(valid, b, N, 1)
print("Windows -> train: {0}, val: {1}, test: {2}".format(
    len(train_starts), len(val_starts), len(test_starts)))
if min(len(train_starts), len(val_starts), len(test_starts)) == 0:
    raise SystemExit("A split has no valid windows. Check the data files / WINDOW_SIZE.")

# Per-subcarrier statistics from training rows only.
train_rows = csi[:a][valid[:a]]
feat_mean = train_rows.mean(axis=0, dtype=np.float64)
feat_std = train_rows.std(axis=0, dtype=np.float64)
feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
del train_rows

y_train = window_mean_hr[train_starts]
y_mean = float(y_train.mean())
y_std = float(max(y_train.std(), 1.0))
print("Target (train): mean {0:.2f} bpm, std {1:.2f} bpm".format(y_mean, y_std))

train_seq = WindowSequence(csi, window_mean_hr, train_starts, shuffle=True, seed=SEED)
val_seq = WindowSequence(csi, window_mean_hr, val_starts, shuffle=False)
test_seq = WindowSequence(csi, window_mean_hr, test_starts, shuffle=False)

FINGERPRINT = hashlib.sha1(json.dumps({
    "window": WINDOW_SIZE, "features": EXPECTED_FEATURES, "hr_range": [HR_MIN, HR_MAX],
    "splits": [TRAIN_FRAC, VAL_FRAC], "arch": [CONV_FILTERS, LSTM_1, LSTM_2, DENSE_UNITS],
    "dropout": [DROPOUT, INPUT_NOISE], "seed": SEED,
    "files": [os.path.getsize(HR_DATA_PATH), os.path.getsize(TRAINING_DATA_PATH)],
}, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------- the model ---------------------------------
def build_model():
    inp = keras.Input(shape=(WINDOW_SIZE, EXPECTED_FEATURES), name="main_input")
    x = keras.layers.Normalization(axis=-1, mean=feat_mean.astype(np.float32),
                                   variance=(feat_std ** 2).astype(np.float32),
                                   name="input_norm")(inp)
    x = keras.layers.GaussianNoise(INPUT_NOISE, name="input_noise")(x)          # train-time only
    x = keras.layers.Conv1D(CONV_FILTERS, 5, padding="same", activation="gelu", name="conv_1")(x)
    x = keras.layers.LayerNormalization(name="ln_1")(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(LSTM_1, return_sequences=True), name="lstm_1")(x)
    x = keras.layers.Dropout(DROPOUT, name="dropout_1")(x)
    x = keras.layers.LSTM(LSTM_2, name="lstm_2")(x)
    x = keras.layers.Dropout(DROPOUT, name="dropout_2")(x)
    x = keras.layers.Dense(DENSE_UNITS, activation="gelu", name="dense_1")(x)
    x = keras.layers.Dense(1, name="dense_out")(x)
    out = keras.layers.Rescaling(scale=y_std, offset=y_mean, name="hr_output")(x)   # -> bpm
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE,
                                         weight_decay=WEIGHT_DECAY, clipnorm=1.0),
        loss=keras.losses.Huber(delta=HUBER_DELTA),
        metrics=["mae"],
    )
    return model


# ------------------------- resumable checkpointing ---------------------------
def atomic_save_model(model, path):
    tmp = path[:-len(".keras")] + ".tmp.keras"
    model.save(tmp)
    os.replace(tmp, path)          # a disconnect mid-write can never corrupt the checkpoint


def atomic_write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def load_resume_state():
    if not (os.path.isfile(STATE_PATH) and os.path.isfile(LATEST_CKPT)):
        return None
    try:
        with open(STATE_PATH) as f:
            st = json.load(f)
    except (ValueError, OSError):
        print("WARNING: could not read {0}; starting fresh.".format(STATE_PATH))
        return None
    if st.get("fingerprint") != FINGERPRINT:
        print("Checkpoint was made with a different data/model configuration; starting fresh.")
        return None
    return st


class ResumableRun(keras.callbacks.Callback):
    """Per epoch: save latest (with optimizer state) and best checkpoints, persist the
    early-stopping counters, and honour the stop file."""

    def __init__(self, state):
        super().__init__()
        self.state = state

    def on_epoch_end(self, epoch, logs=None):
        cur = (logs or {}).get(MONITOR)
        if cur is None or not np.isfinite(cur):
            print("Epoch {0}: {1} is not finite; stopping without touching checkpoints.".format(
                epoch + 1, MONITOR))
            self.model.stop_training = True
            return
        st = self.state
        if cur < st["best"] - MIN_DELTA:
            st.update(best=float(cur), best_epoch=epoch, wait=0)
            atomic_save_model(self.model, BEST_MODEL_PATH)
        else:
            st["wait"] += 1
        st["epoch"] = epoch
        atomic_save_model(self.model, LATEST_CKPT)
        atomic_write_json(st, STATE_PATH)
        if st["wait"] >= PATIENCE:
            print("Early stopping: no {0} improvement for {1} epochs (best epoch {2}).".format(
                MONITOR, PATIENCE, st["best_epoch"] + 1))
            self.model.stop_training = True
        elif os.path.exists(TRAIN_STOP_FILE_PATH):
            print("Stop file found; stopping after epoch {0}.".format(epoch + 1))
            self.model.stop_training = True


# ----------------------------------- train -----------------------------------
if os.path.exists(TRAIN_STOP_FILE_PATH):
    os.remove(TRAIN_STOP_FILE_PATH)
    print("Removed stale stop file.")

state = load_resume_state()
if state is not None:
    print("Found checkpoint, auto-resuming from {0}".format(LATEST_CKPT))
    model = keras.models.load_model(LATEST_CKPT)           # weights + optimizer + current LR
    initial_epoch = state["epoch"] + 1
else:
    model = build_model()
    state = {"fingerprint": FINGERPRINT, "epoch": -1, "best": float("inf"),
             "best_epoch": -1, "wait": 0}
    initial_epoch = 0

print("Resuming at epoch {0}".format(initial_epoch))
model.summary()

if initial_epoch >= TOTAL_EPOCHS or state["wait"] >= PATIENCE:
    print("Training already finished (epoch {0} of {1}, patience counter {2}/{3}).".format(
        initial_epoch, TOTAL_EPOCHS, state["wait"], PATIENCE))
else:
    model.fit(
        train_seq,
        epochs=TOTAL_EPOCHS,
        initial_epoch=initial_epoch,
        validation_data=val_seq,
        verbose=2,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(monitor=MONITOR, factor=0.5, patience=5,
                                              min_lr=1e-5, verbose=1),
            ResumableRun(state),
        ],
    )

# --------------------------- evaluate the BEST epoch ---------------------------
if not os.path.isfile(BEST_MODEL_PATH):
    raise SystemExit("No best checkpoint was written (training diverged on epoch 1?).")
best = keras.models.load_model(BEST_MODEL_PATH)


def report(name, seq):
    y = seq.y_true()
    p = best.predict(seq, verbose=0).ravel()
    err = p - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out = {
        "mae_bpm": float(np.mean(np.abs(err))),
        "rmse_bpm": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "pearson_r": float(np.corrcoef(y, p)[0, 1]) if y.std() > 0 and p.std() > 0 else float("nan"),
        # Reference: always predicting the training-set mean HR. The model must beat this.
        "baseline_mae_bpm": float(np.mean(np.abs(y - y_mean))),
        "baseline_rmse_bpm": float(np.sqrt(np.mean((y - y_mean) ** 2))),
    }
    print("{0:>5}: MAE {mae_bpm:.2f} bpm | RMSE {rmse_bpm:.2f} | R2 {r2:.3f} | r {pearson_r:.3f}"
          "   (predict-the-mean baseline: MAE {baseline_mae_bpm:.2f}, RMSE {baseline_rmse_bpm:.2f})"
          .format(name, **out))
    return out


metrics = {"val": report("val", val_seq), "test": report("test", test_seq),
           "best_epoch": state["best_epoch"] + 1, "window_size": WINDOW_SIZE}
atomic_write_json(metrics, METRICS_PATH)
atomic_save_model(best, MODEL_SAVE_PATH)
print("Saved best model to {0}. Model training complete!".format(MODEL_SAVE_PATH))