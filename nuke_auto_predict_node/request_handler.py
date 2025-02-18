import json
import logging
import urllib.request
from urllib.error import URLError, HTTPError

from contextlib import contextmanager
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)

_g_request_handler = None


def get_request_handler():
    global _g_request_handler
    if _g_request_handler is None:
        _g_request_handler = RequestHandler(f"http://127.0.0.1:8000/")
    return _g_request_handler


class NukeRequestError(Exception):
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
    def __init__(self, base_url, timeout=1):
        self.base_url = base_url
        self.timeout = timeout

    @contextmanager
    def _handle_request_error(self):
        try:
            yield
        except (URLError, HTTPError) as e:
            error_body = None
            if hasattr(e, "read"):
                error_body = e.read().decode("utf-8")
            status_code = getattr(e, "code", None)

            raise NukeRequestError(
                f"Request failed: {str(error_body)}",
                status_code=status_code,
                response_body=error_body,
            )

        except Exception as e:
            raise NukeRequestError(f"Unexpected error: {str(e)}")

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

    def kickoff_training(self, file_paths):
        data = {
            "file_paths": file_paths,
        }

        log.info("Posted to train.")
        return self.post("train", data, custom_timeout=3600)
