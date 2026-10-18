import os


class RequestHandler:
    def __init__(self):
        self.timeout = 1
        self.configure("127.0.0.1", os.environ.get("AUTO_PREDICT_PORT", "8080"))

    def configure(self, host, port):
        if not host.startswith("http"):
            host = f"http://{host}"
        self.port = port
        self.base_url = f"{host}:{port}/"
