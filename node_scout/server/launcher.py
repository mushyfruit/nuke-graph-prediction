import os
import sys
import atexit
import signal
import subprocess

from ..logging_config import get_logger
from .constants import DirectoryConfig

log = get_logger(__name__)

_inference_launcher = None


def get_inference_launcher():
    global _inference_launcher
    if _inference_launcher is None:
        _inference_launcher = InferenceLauncher()
    return _inference_launcher


class InferenceLauncher:
    def __init__(self, venv_path=None):
        self.venv_path = venv_path if venv_path else DirectoryConfig.VENV_DIR
        self.python_path = os.path.join(self.venv_path, "bin", "python3")
        self.process = None

        atexit.register(self.stop_service)

    def start_service(self, host=None, port=None):
        if self.process is not None and self.process.poll() is None:
            return True
        env = self.get_subprocess_env()

        port = port or os.environ.get("AUTO_PREDICT_PORT", "8080")
        host = host or "127.0.0.1"

        cmd = [
            str(self.python_path),
            DirectoryConfig.INFERENCE_SCRIPT_PATH,
            str(host),
            str(port),
        ]
        if not os.path.isfile(os.path.join(self.venv_path, ".nodescout-ready")):
            cmd = [
                DirectoryConfig.VENV_SETUP_SCRIPT,
                env["LD_LIBRARY_PATH"],
                self.venv_path,
                *cmd,
            ]
        # prctl(PR_SET_PDEATHSIG, SIGKILL): kill the server even if Nuke skips atexit.
        cmd = [
            os.path.join(env["LD_LIBRARY_PATH"], "python3.11"),
            "-c",
            "import ctypes, os, sys; ctypes.CDLL(None).prctl(1, 9); "
            "os.execv(sys.argv[1], sys.argv[1:])",
            *cmd,
        ]

        log.info(
            "Starting inference service; setup and server output: %s",
            DirectoryConfig.SERVER_LOG_FILE,
        )
        with open(DirectoryConfig.SERVER_LOG_FILE, "w") as log_file:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )

        return True

    def stop_service(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=2)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                log.warning("Force-killing inference subprocess.")
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()

            self.process = None

    def get_subprocess_env(self):
        env = os.environ.copy()
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTORCH_INSTALL"):
            env.pop(name, None)
        interpreter = (
            self.python_path if os.path.exists(self.python_path) else sys.executable
        )
        env["LD_LIBRARY_PATH"] = os.path.dirname(os.path.realpath(interpreter))
        env["PYTHONNOUSERSITE"] = "1"
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
        env["TORCH_COMPILE_DISABLE"] = "1"
        return env

    def restart_service(self, host=None, port=None):
        self.stop_service()
        return self.start_service(host=host, port=port)


def launch_inference_service():
    get_inference_launcher().start_service()
