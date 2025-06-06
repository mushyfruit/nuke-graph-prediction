import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any

VOCAB = "vocab.json"
MODEL_NAME = "nuke_predictor_gat"


class DirectoryConfig:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    MODEL_PATH = os.path.join(BASE_DIR, "checkpoints")
    MODEL_DATA_FOLDER = os.path.join(BASE_DIR, "tmp", "nuke_graphs")
    DATA_CACHE_PATH = os.path.join(MODEL_PATH, "data_cache")
    VOCAB_PATH = os.path.join(DATA_CACHE_PATH, VOCAB)
    TRAINING_LOG_FILE = os.path.join(MODEL_PATH, "logs", "training.log")


class NukeScript:
    ROOT = "root"
    NODES = "nodes"


class TrainingPhase(Enum):
    IDLE = "idle"
    SERIALIZING = "serializing"
    SERIALIZATION_COMPLETE = "serialization_complete"
    TRAINING = "training"
    COMPLETE = "complete"
    ERROR = "error"
    NO_CHANGE = "no_change"


@dataclass
class TrainingStatus:
    phase: TrainingPhase
    progress: Optional[float] = None
    label: Optional[str] = None
    error: Optional[str] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    training_loss: Optional[float] = None
    training_accuracy: Optional[float] = None
    validation_accuracy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        status_dict = {"status": self.phase.value}

        if self.progress:
            status_dict["progress"] = self.progress

        if self.label:
            status_dict["label"] = self.label

        if self.error:
            status_dict["error"] = self.error

        if self.phase == TrainingPhase.TRAINING:
            status_dict.update(
                {
                    "current_epoch": self.current_epoch,
                    "total_epochs": self.total_epochs,
                    "training_loss": self.training_loss,
                    "training_accuracy": self.training_accuracy,
                    "validation_accuracy": self.validation_accuracy,
                }
            )

        return status_dict
