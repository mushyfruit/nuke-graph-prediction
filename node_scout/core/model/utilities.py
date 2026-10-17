import os
import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from .constants import DirectoryConfig, MODEL_NAME

if TYPE_CHECKING:
    from .gnn.gat import NukeGATPredictor

log = logging.getLogger(__name__)


def check_for_model_on_disk(model_name: Optional[str] = None) -> bool:
    if model_name is None:
        model_name = MODEL_NAME

    model_checkpoint_path = os.path.join(
        DirectoryConfig.MODEL_PATH, f"{model_name}_model.pt"
    )
    return os.path.exists(model_checkpoint_path)


def save_model_checkpoint(model: "NukeGATPredictor", model_name: str):
    import torch

    os.makedirs(DirectoryConfig.MODEL_PATH, exist_ok=True)

    model_path = os.path.join(DirectoryConfig.MODEL_PATH, f"{model_name}_model.pt")
    torch.save(model.to_checkpoint(), model_path)
    log.info(f"Model saved to {DirectoryConfig.MODEL_PATH}")


def check_state_dict(state_dict: Dict[str, Any], unwanted_prefixes: List[str] = None):
    if unwanted_prefixes is None:
        unwanted_prefixes = ["_orig_mod.", "module."]

    for k, v in list(state_dict.items()):
        for unwanted_prefix in unwanted_prefixes:
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    return state_dict
