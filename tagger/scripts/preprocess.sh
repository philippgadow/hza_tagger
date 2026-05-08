#!/usr/bin/env bash
# Generate the normalisation dictionary from the training H5 file.
# This must be run before the first training.
#
# SALT 0.11 does not have a `salt preprocess` subcommand; we use a
# standalone script instead.

set -euo pipefail

# Resolve Python from the active conda env
PYTHON="${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}"
PYTHON="${PYTHON:-$(command -v python3 2>/dev/null || command -v python)}"

TRAIN_FILE=${1:-data/train.h5}
# fall back to test_out.h5 if train.h5 hasn't been created yet
[[ -f "${TRAIN_FILE}" ]] || TRAIN_FILE="$(ls data/*.h5 2>/dev/null | head -1 || true)"
[[ -f "${TRAIN_FILE:-}" ]] || { echo "ERROR: no H5 file found in data/. Run the converter first."; exit 1; }
VARIABLES=tagger/configs/hza_variables.yaml
NORM_DICT=tagger/configs/norm_dict.yaml

echo "==> Computing normalisation statistics from ${TRAIN_FILE} …"
"${PYTHON}" tagger/scripts/create_norm_dict.py \
    --input   "${TRAIN_FILE}" \
    --config  "${VARIABLES}" \
    --output  "${NORM_DICT}"

echo "Norm dict written to ${NORM_DICT}"
