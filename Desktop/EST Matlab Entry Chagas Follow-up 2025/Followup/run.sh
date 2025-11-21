#!/bin/bash
set -e

# --- Check input arguments ---
if [ "\$#" -lt 2 ]; then
    echo "Usage: \ <input_directory> <output_directory> [verbose]"
    exit 1
fi

INPUT_DIR="\"
OUTPUT_DIR="\"
VERBOSE="\"  # Default verbose level = 1

# --- Run the TRAIN model if executable exists ---
if [ -x "./team_train_model.exe" ]; then
    echo "Running TRAIN model..."
    ./team_train_model.exe "\" "\" "\"
    exit 0
fi

# --- Run the RUN model if executable exists ---
if [ -x "./team_run_model.exe" ]; then
    echo "Running RUN model..."
    ./team_run_model.exe "\" "\" "\"
    exit 0
fi

# --- Error if no compiled model found ---
echo "ERROR: No compiled model found! Please ensure team_train_model.exe or team_run_model.exe is present."
exit 1
