import os
import logging
from typing import TYPE_CHECKING

from .constants import DirectoryConfig, MODEL_NAME

if TYPE_CHECKING:
    from .gnn.gat import NukeGATPredictor

log = logging.getLogger(__name__)


def check_for_model_on_disk(model_name: str = MODEL_NAME) -> bool:
    model_checkpoint_path = os.path.join(
        DirectoryConfig.MODEL_PATH, f"{model_name}_model.pt"
    )
    return os.path.exists(model_checkpoint_path)


def save_model_checkpoint(model: "NukeGATPredictor", model_name: str, vocabulary):
    import torch

    os.makedirs(DirectoryConfig.MODEL_PATH, exist_ok=True)

    model_path = os.path.join(DirectoryConfig.MODEL_PATH, f"{model_name}_model.pt")
    checkpoint = model.to_checkpoint()
    checkpoint["vocabulary"] = [vocabulary.get_type(i) for i in range(len(vocabulary))]
    torch.save(checkpoint, model_path + ".tmp")
    os.replace(model_path + ".tmp", model_path)
    log.info(f"Model saved to {DirectoryConfig.MODEL_PATH}")
