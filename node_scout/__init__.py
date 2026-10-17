import logging
import os
import sys
from .core.model.constants import DirectoryConfig

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

logger = logging.getLogger("node_scout")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    os.makedirs(os.path.dirname(DirectoryConfig.LOG_FILE), exist_ok=True)
    for handler in (
        logging.FileHandler(DirectoryConfig.LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)


def on_startup(*args, **kwargs):
    from .nuke_startup import on_startup as _on_startup

    return _on_startup(*args, **kwargs)
