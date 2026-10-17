import logging
import os
import sys
import atexit
import signal
import subprocess
import time
from functools import lru_cache

from ..core.model.constants import DirectoryConfig

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_inference_launcher():
    return InferenceLauncher()


class InferenceLauncher:
    def __init__(self, venv_path=None):
        self.venv_path = venv_path if venv_path else DirectoryConfig.VENV_DIR
        self.python_path = os.path.join(self.venv_path, "bin", "python3")
        self.process = None
        self._restart_timer = None

        atexit.register(self.stop_service)

    def start_service(self, host=None, port=None):
        if self.process is not None and self.process.poll() is None:
            return
        env = self.get_subprocess_env()

        port = port or os.environ.get("AUTO_PREDICT_PORT", "8080")
        host = host or "127.0.0.1"

        cmd = [
            str(self.python_path),
            DirectoryConfig.INFERENCE_SCRIPT_PATH,
            str(host),
            str(port),
        ]
        needs_setup = not os.path.isfile(os.path.join(self.venv_path, ".nodescout-ready"))
        if needs_setup:
            cmd = [
                DirectoryConfig.VENV_SETUP_SCRIPT,
                env["LD_LIBRARY_PATH"],
                self.venv_path,
                *cmd,
            ]
        # During setup, supervise the entire process group, including pip.
        wrapper = (
            "import ctypes, os, signal, subprocess, sys; "
            "signal.signal(signal.SIGTERM, lambda *_: os.killpg(0, signal.SIGKILL)); "
            "ctypes.CDLL(None).prctl(1, signal.SIGTERM); "
            f"os.getppid() == {os.getpid()} or os.killpg(0, signal.SIGKILL); "
            "sys.exit(subprocess.call(sys.argv[1:]))"
            if needs_setup else
            "import ctypes, os, sys; ctypes.CDLL(None).prctl(1, 9); "
            f"os.getppid() == {os.getpid()} or os.kill(os.getpid(), 9); "
            "os.execv(sys.argv[1], sys.argv[1:])"
        )
        cmd = [os.path.join(env["LD_LIBRARY_PATH"], "python3.11"), "-c", wrapper, *cmd]

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

    def stop_service(self):
        if self._restart_timer is not None:
            self._restart_timer.stop()
            self._restart_timer.deleteLater()
            self._restart_timer = None
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
        if self._restart_timer is not None:
            return
        try:
            from PySide6 import QtCore
        except ImportError:
            from PySide2 import QtCore

        process = self.process
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 2

        def restart_when_stopped():
            if process is not None and process.poll() is None:
                if time.monotonic() >= deadline:
                    os.killpg(process.pid, signal.SIGKILL)
                return
            self._restart_timer.stop()
            self._restart_timer.deleteLater()
            self._restart_timer = None
            self.process = None
            self.start_service(host=host, port=port)

        self._restart_timer = QtCore.QTimer()
        self._restart_timer.timeout.connect(restart_when_stopped)
        self._restart_timer.start(25)
