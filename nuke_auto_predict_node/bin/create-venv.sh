#!/bin/bash
set -e

# Create and activate venv
python3 -m venv inference_venv
source inference_venv/bin/activate

# Upgrade pip
inference_venv/bin/python3 -m pip install --upgrade pip
inference_venv/bin/python3 -m pip install -r ../requirements.txt

# Deactivate at the end
deactivate

echo "Nuke Auto-Predict setup completed successfully"