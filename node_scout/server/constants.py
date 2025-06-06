import os


class DirectoryConfig:
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    VENV_DIR_NAME = os.getenv("VENV_DIR_NAME", "inference_venv")
    INFERENCE_SCRIPT_NAME = os.getenv("INFERENCE_SCRIPT_NAME", "inference.py")

    VENV_DIR = os.path.join(BASE_DIR, VENV_DIR_NAME)
    VENV_SETUP_SCRIPT = os.path.join(BASE_DIR, "bin", "create-venv.sh")
    INFERENCE_SCRIPT_PATH = os.path.join(BASE_DIR, "server", INFERENCE_SCRIPT_NAME)
    LOG_FILE = os.path.join(BASE_DIR, "checkpoints", "logs", "auto-predict.log")
    SERVER_LOG_FILE = os.path.join(BASE_DIR, "checkpoints", "logs", "server.log")
