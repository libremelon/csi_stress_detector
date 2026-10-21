#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combine_datasets.py

Merge several per-condition CSI/HR datasets into a single dataset that can be
fed straight into the training pipeline (the same format the collection script
produces with --collect-training-data).

Expected input layout (one folder per condition, any number of folders):

    exerted/
        training_data.txt    # one CSI sample per line (comma separated features)
        hr_data.txt          # one heart-rate value per line
    normal/
        training_data.txt
        hr_data.txt
    ...

Output:

    combined/
        training_data.txt    # all CSI rows, in folder order
        hr_data.txt          # all HR rows, same order, 1:1 with CSI rows

Rows are concatenated in the order the folders appear in DATASET_DIRS, so
CSI row N in the output always corresponds to HR row N.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Configuration  (edit DATASET_DIRS to add/remove folders)
# ---------------------------------------------------------------------------
DATASET_DIRS = [
    "datasets/exertion",
    "datasets/still",
    "datasets/low-cortisol",
    "datasets/walk"
]

OUTPUT_DIR = "datasets/combined"

CSI_FILENAME = "training_data.txt"
HR_FILENAME = "hr_data.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_and_width(path):
    """Return (number of non-empty lines, feature count of the first line)."""
    n = 0
    width = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            n += 1
            if width is None:
                width = s.count(",") + 1
    return n, width


def iter_nonempty_lines(path):
    """Yield stripped, non-empty lines from a file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s


def validate(dataset_dirs):
    """Check every folder before writing anything. Returns a plan list."""
    plan = []
    feature_count = None

    for folder in dataset_dirs:
        csi_path = os.path.join(folder, CSI_FILENAME)
        hr_path = os.path.join(folder, HR_FILENAME)

        for p in (csi_path, hr_path):
            if not os.path.isfile(p):
                sys.exit(f"ERROR: missing file: {p}")

        n_csi, width = count_and_width(csi_path)
        n_hr, _ = count_and_width(hr_path)

        if n_csi != n_hr:
            sys.exit(
                f"ERROR: {folder} is misaligned: "
                f"{n_csi} CSI rows vs {n_hr} HR rows"
            )

        if n_csi == 0:
            print(f"WARNING: {folder} is empty - skipping it.", file=sys.stderr)
            continue

        if feature_count is None:
            feature_count = width
        elif width != feature_count:
            sys.exit(
                f"ERROR: {folder} has {width} features per CSI row, "
                f"but a previous folder has {feature_count}. "
                "These datasets cannot be concatenated."
            )

        plan.append((folder, n_csi))

    if not plan:
        sys.exit("ERROR: no usable samples found in any dataset folder.")

    return plan, feature_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    plan, feature_count = validate(DATASET_DIRS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csi_path = os.path.join(OUTPUT_DIR, CSI_FILENAME)
    out_hr_path = os.path.join(OUTPUT_DIR, HR_FILENAME)

    total = 0
    with open(out_csi_path, "w", encoding="utf-8") as out_csi, \
         open(out_hr_path, "w", encoding="utf-8") as out_hr:

        for folder, n_samples in plan:
            csi_path = os.path.join(folder, CSI_FILENAME)
            hr_path = os.path.join(folder, HR_FILENAME)

            for csi_line, hr_line in zip(
                iter_nonempty_lines(csi_path),
                iter_nonempty_lines(hr_path),
            ):
                out_csi.write(csi_line + "\n")
                out_hr.write(hr_line + "\n")

            total += n_samples
            print(f"  {folder:<20s} -> {n_samples:>7d} samples merged")

    print()
    print(f"Wrote {total} samples ({feature_count} features each)")
    print(f"  {out_csi_path}")
    print(f"  {out_hr_path}")


if __name__ == "__main__":
    main()