import os
import json
import urllib.request
from urllib.error import URLError, HTTPError

from contextlib import contextmanager
from typing import Dict, Any, Optional, List

from ..logging_config import get_logger

log = get_logger(__name__)


class InferenceRequestError(Exception):
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(self.message)


class RequestHandler:
    _instance = None
    _initialized = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self, host: Optional[str] = None, port: Optional[int] = None, timeout: int = 1
    ):
        if self._initialized:
            return

        self.timeout = timeout
        self.host = host or "http://127.0.0.1"
        self.port = port or os.environ.get("AUTO_PREDICT_PORT", "8080")

        self.base_url = f"{self.host}:{self.port}/"

        log.debug(f"Starting the RequestHandler for {self.base_url}")
        self._initialized = True

    def set_host(self, host):
        if not host.startswith("http"):
            host = f"http://{host}"

        self.host = host
        self._update_base_url()

    def set_port(self, port):
        self.port = port
        self._update_base_url()

    def _update_base_url(self):
        self.base_url = f"{self.host}:{self.port}/"

    @contextmanager
    def _handle_request_error(self):
        try:
            yield
        except (URLError, HTTPError) as e:
            error_body = None
            if hasattr(e, "read"):
                error_body = e.read().decode("utf-8")
            status_code = getattr(e, "code", None)

            raise InferenceRequestError(
                f"Request failed: {str(error_body)}",
                status_code=status_code,
                response_body=error_body,
            )

        except Exception as e:
            raise InferenceRequestError(f"Unexpected error: {str(e)}")

    def post(
        self,
        endpoint: str,
        data: Dict[str, Any],
        custom_timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"

        encoded_data = json.dumps(data).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        request = urllib.request.Request(
            url, data=encoded_data, headers=headers, method="POST"
        )

        with self._handle_request_error():
            timeout = custom_timeout if custom_timeout is not None else self.timeout
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = response.read().decode("utf-8")
                return json.loads(response_body)

    def get(
        self,
        endpoint: str,
        custom_timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"

        headers = {"Accept": "application/json"}
        request = urllib.request.Request(url, headers=headers, method="GET")

        with self._handle_request_error():
            timeout = custom_timeout if custom_timeout is not None else self.timeout
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = response.read().decode("utf-8")
                return json.loads(response_body)

    def kickoff_training(
        self, file_paths: List[str], memory_allocation: float, enable_fine_tuning: bool
    ):
        data = {
            "file_paths": file_paths,
            "memory_allocation": memory_allocation,
            "enable_fine_tuning": enable_fine_tuning,
        }

        log.info("Posted to train.")
        return self.post("train", data, custom_timeout=3600)
