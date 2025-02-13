import os
import sys
import atexit
import signal
import subprocess
import logging

from .constants import VENV_DIR_PATH, VENV_SETUP_SCRIPT, INFERENCE_SCRIPT_PATH

log = logging.getLogger(__name__)


def create_launcher_venv():
    subprocess.run(VENV_SETUP_SCRIPT, cwd=os.path.dirname(__file__))


class InferenceLauncher:
    def __init__(self, venv_path=None):
        self.venv_path = venv_path if venv_path else VENV_DIR_PATH
        self.python_path = os.path.join(self.venv_path, "bin", "python3")
        self.process = None

        atexit.register(self.stop_service)
        self.setup_signal_handlers()

    def start_service(self, port=8000):
        env = self.get_subprocess_env()

        cmd = [str(self.python_path), INFERENCE_SCRIPT_PATH, str(port)]

        log.info(
            f"Started the inference subprocess..."
        )
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Thread-safe alternative to preexec_fn=os.setsid
            start_new_session=True,
            #cwd=os.path.dirname(os.path.abspath(script_path)),
        )

        # for line in self.process.stdout:
        #     log.info(f"Subprocess stdout: {line.strip()}")
        # for line in self.process.stderr:
        #     log.info(f"Subprocess stderr: {line.strip()}")

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
        log.info("Terminated the subprocess...")
        if self.process is not None:
            self.process.terminate()
            self.process = None

    def get_subprocess_env(self):
        env = os.environ.copy()
        python_path = os.path.join(
            self.venv_path,
            "lib",
            "python3.12",
            "site-packages",
        )

        # TODO: Formalize how we retrieve the PyTorch install.
        base_path = "/home/andrew/Documents/Dev/machine_learning_project/.venv"
        torch64_path = os.path.join(base_path, "lib64", "python3.12", "site-packages")
        torch_path = os.path.join(base_path, "lib", "python3.12", "site-packages")

        env["PYTHONPATH"] = f"{python_path}:{torch_path}:{torch64_path}"

        return env


def launch_inference_service():
    # Ensure the inference service's virtual env exists.
    log.info('launch')
    if not os.path.exists(VENV_DIR_PATH):
        create_launcher_venv()
        return

    launcher = InferenceLauncher()
    launcher.start_service()
