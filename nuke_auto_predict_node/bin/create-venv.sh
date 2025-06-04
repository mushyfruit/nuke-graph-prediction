#!/bin/bash
set -e

# Passed in by the subprocess.
PARENT_DIR="$1"

# Locate Nuke's bundled python to ensure the venv matches Nuke's python version.
PYTHON_BIN=$(find "$PARENT_DIR" \( -type f -o -type l \) -name "python3" -executable | head -n 1)
if [ -z "$PYTHON_BIN" ]; then
  echo "Error: Unable to locate Nuke's bundled Python executable."
  echo "Error: Exiting without starting the inference server."
  exit 1
fi

# Create and activate venv
VENV_DIR=$(realpath inference_venv)
$PYTHON_BIN -m venv "$VENV_DIR"
source inference_venv/bin/activate

# Get an absolute path to the inference requirements.
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
REQ_FILE="$SCRIPT_DIR/../../inference_requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
  echo "Error: Requirements file not found at $REQ_FILE"
  exit 1
fi

# Install requirements (excluding pytorch)
VENV_PYTHON="$VENV_DIR/bin/python3"
$VENV_PYTHON -m pip install --upgrade pip
$VENV_PYTHON -m pip install -r "$REQ_FILE"

# Check if PYTORCH_PATH is defined; if not, install torch
if [ -z "${PYTORCH_PATH:-}" ]; then
  echo "PYTORCH_PATH not set — installing PyTorch..."
  $VENV_PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cu121
  $VENV_PYTHON -m pip install torch-geometric
else
  echo "PYTORCH_PATH is set — skipping torch installation."
fi

# Deactivate at the end
deactivate

echo "Nuke Auto-Predict setup completed successfully"