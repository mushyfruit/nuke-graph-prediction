import queue
import logging
import threading

from typing import Optional
from .constants import TrainingStatus

log = logging.getLogger(__name__)


class StatusQueue(queue.Queue):
    def __init__(self, maxsize=100):
        super(StatusQueue, self).__init__(maxsize=maxsize)
        self._lock = threading.Lock()

    def safe_put(self, status: TrainingStatus) -> bool:
        with self._lock:
            try:
                self.put(status, block=False)
                return True
            except queue.Full:
                try:
                    self.get_nowait()
                    self.put(status, block=False)
                    return True
                except (queue.Empty, queue.Full):
                    log.warning("Error: failed to put status to queue!")
                    return False

    def safe_get(self) -> Optional[TrainingStatus]:
        try:
            return self.get_nowait()
        except queue.Empty:
            return None

    def get_latest(self) -> Optional[TrainingStatus]:
        try:
            return list(self.queue)[-1]
        except IndexError:
            return None
