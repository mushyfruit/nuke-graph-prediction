import os

VENV_DIR_NAME = "inference_venv"
INFERENCE_SCRIPT_NAME = "inference.py"

BASE_DIR = os.path.dirname(__file__)
VENV_DIR_PATH = os.path.join(BASE_DIR, VENV_DIR_NAME)
VENV_SETUP_SCRIPT = os.path.join(BASE_DIR, "bin", "create-venv.sh")
INFERENCE_SCRIPT_PATH = os.path.join(BASE_DIR, INFERENCE_SCRIPT_NAME)
