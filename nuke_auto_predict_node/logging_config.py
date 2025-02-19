import logging
import sys
from .server.constants import DirectoryConfig

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

logger = logging.getLogger()
while logger.hasHandlers():
    logger.removeHandler(logger.handlers[0])

file_handler = logging.FileHandler(DirectoryConfig.LOG_FILE, mode="w")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.propagate = True


def get_logger(name=None):
    """Returns a logger instance with the given module name."""
    child_logger = logging.getLogger(name if name else __name__)
    child_logger.setLevel(logging.INFO)
    child_logger.propagate = True
    return child_logger


# Log initialization
logger.info(f"Logging initialized for {__name__}")
