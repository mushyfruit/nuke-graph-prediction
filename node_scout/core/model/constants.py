import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any

MODEL_NAME = "nuke_predictor_gat"


class DirectoryConfig:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    MODEL_PATH = os.path.join(BASE_DIR, "checkpoints")
    MODEL_DATA_FOLDER = os.path.join(BASE_DIR, "tmp", "nuke_graphs")
    TMP_GRAPH_FOLDER = os.path.join(BASE_DIR, "tmp", "graph_vis")
    DATA_CACHE_PATH = os.path.join(MODEL_PATH, "data_cache")
    VOCAB_PATH = os.path.join(DATA_CACHE_PATH, "vocab.json")

    LOSS_PLOT = os.path.join(TMP_GRAPH_FOLDER, "loss.png")
    TOPK_PLOT = os.path.join(TMP_GRAPH_FOLDER, "topk.png")

    VENV_DIR = os.path.join(BASE_DIR, "inference_venv")
    VENV_SETUP_SCRIPT = os.path.join(BASE_DIR, "bin", "create-venv.sh")
    INFERENCE_SCRIPT_PATH = os.path.join(BASE_DIR, "server", "inference.py")
    LOG_FILE = os.path.join(MODEL_PATH, "logs", "auto-predict.log")
    SERVER_LOG_FILE = os.path.join(MODEL_PATH, "logs", "server.log")


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
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    topk_accuracy: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        status_dict = {"status": self.phase.value}

        if self.progress is not None:
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
                    "validation_loss": self.validation_loss,
                    "validation_accuracy": self.validation_accuracy,
                    "topk_accuracy": self.topk_accuracy,
                }
            )

        return status_dict
