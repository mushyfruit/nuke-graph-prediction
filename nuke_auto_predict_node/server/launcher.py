import os
import atexit
import signal
import subprocess
import logging

from .constants import DirectoryConfig

log = logging.getLogger(__name__)


def create_launcher_venv():
    subprocess.run(DirectoryConfig.VENV_SETUP_SCRIPT, cwd=os.path.dirname(__file__))


class InferenceLauncher:
    def __init__(self, venv_path=None):
        self.venv_path = venv_path if venv_path else DirectoryConfig.VENV_DIR
        self.python_path = os.path.join(self.venv_path, "bin", "python3")
        self.process = None

        atexit.register(self.stop_service)
        self.setup_signal_handlers()

    def start_service(self, port=8000):
        env = self.get_subprocess_env()

        cmd = [str(self.python_path), DirectoryConfig.INFERENCE_SCRIPT_PATH, str(port)]

        log.info(f"Started the inference subprocess...")
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Thread-safe alternative to preexec_fn=os.setsid
            start_new_session=True,
        )

        for line in self.process.stdout:
            log.info(f"Subprocess stdout: {line.strip()}")
        for line in self.process.stderr:
            log.info(f"Subprocess stderr: {line.strip()}")

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

        result = subprocess.run(
            [f"{self.venv_path}/bin/python3", "-V"], capture_output=True, text=True
        )

        version = result.stdout.strip().split()[1]
        major, minor, _ = version.split(".")
        python_dir = f"python{major}.{minor}"

        python_path = os.path.join(self.venv_path, "lib", python_dir, "site-packages")

        paths = [python_path]
        torch_location = os.getenv("PYTORCH_INSTALL")
        if torch_location:
            target = "lib64" if "lib64" in torch_location else "lib"
            for lib_dir in ["lib", "lib64"]:
                torch_path = torch_location.replace(target, lib_dir)
                if os.path.exists(torch_path):
                    paths.append(torch_path)

        env["PYTHONPATH"] = ":".join(paths)
        return env


def launch_inference_service():
    # Ensure the inference service's virtual env exists.
    if not os.path.exists(DirectoryConfig.VENV_DIR):
        create_launcher_venv()
        return

    launcher = InferenceLauncher()
    launcher.start_service()
