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


def create_launcher_venv():
    nuke_python_parent_dir = os.path.dirname(sys.executable)
    subprocess.run(
        [DirectoryConfig.VENV_SETUP_SCRIPT, nuke_python_parent_dir],
        cwd=DirectoryConfig.BASE_DIR,
    )


class InferenceLauncher:
    def __init__(self, venv_path=None):
        self.venv_path = venv_path if venv_path else DirectoryConfig.VENV_DIR
        self.python_path = os.path.join(self.venv_path, "bin", "python3")
        self.process = None

        atexit.register(self.stop_service)
        self.setup_signal_handlers()

    def start_service(self, host=None, port=None):
        env = self.get_subprocess_env()

        port = port or os.environ.get("AUTO_PREDICT_PORT", "8080")
        host = host or "127.0.0.1"

        cmd = [
            str(self.python_path),
            DirectoryConfig.INFERENCE_SCRIPT_PATH,
            str(host),
            str(port),
        ]

        log.debug("Started the inference subprocess...")
        with open(DirectoryConfig.SERVER_LOG_FILE, "a") as log_file:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=log_file,
                # Thread-safe alternative to preexec_fn=os.setsid
                start_new_session=True,
            )

        return True

    def setup_signal_handlers(self):
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT):
            signal.signal(sig, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        if signum == signal.SIGQUIT:
            os.killpg(self.process.pid, signal.SIGTERM)
        else:
            # For other signals, try graceful shutdown first
            self.stop_service()

    def stop_service(self):
        log.info("Terminated the inference subprocess...")
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                log.warning("Force-killing inference subprocess.")
                self.process.kill()

            self.process = None

    def get_subprocess_env(self):
        env = os.environ.copy()

        result = subprocess.run(
            [f"{self.venv_path}/bin/python3", "-V"], capture_output=True, text=True
        )

        version = result.stdout.strip().split()[1]
        major, minor, _ = version.split(".")
        python_dir = f"python{major}.{minor}"

        python_path = os.path.join(self.venv_path, "lib", python_dir, "site-packages")
        python_path_64 = os.path.join(
            self.venv_path, "lib64", python_dir, "site-packages"
        )

        paths = [python_path, python_path_64]

        torch_location = os.getenv("PYTORCH_INSTALL")
        if torch_location:
            target = "lib64" if "lib64" in torch_location else "lib"
            for lib_dir in ["lib", "lib64"]:
                torch_path = torch_location.replace(target, lib_dir)
                if os.path.exists(torch_path):
                    paths.append(torch_path)

        env["PYTHONPATH"] = ":".join(paths)
        return env

    def restart_service(self, host=None, port=None):
        self.stop_service()
        return self.start_service(host=host, port=port)


def launch_inference_service():
    # Ensure the inference service's virtual env exists.
    if not os.path.exists(DirectoryConfig.VENV_DIR):
        create_launcher_venv()

    launcher = get_inference_launcher()
    launcher.start_service()
