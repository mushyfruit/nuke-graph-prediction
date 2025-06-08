import os
import json
import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from .constants import DirectoryConfig, MODEL_NAME

if TYPE_CHECKING:
    from .gat import NukeGATPredictor

log = logging.getLogger(__name__)


def check_for_model_on_disk(model_name: Optional[str] = None) -> bool:
    if model_name is None:
        model_name = MODEL_NAME

    model_checkpoint_path = os.path.join(
        DirectoryConfig.MODEL_PATH, f"{model_name}_model.pt"
    )
    return os.path.exists(model_checkpoint_path)


def json_back_to_nk():
    import glob

    current_dir = os.path.dirname(os.path.dirname(__file__))
    source_dir = os.path.join(current_dir, "test_scripts")

    source_data = {}

    # Create the pattern to match all JSON files
    json_pattern = os.path.join(source_dir, "*.json")

    json_files = glob.glob(json_pattern)
    if not json_files:
        log.info(f"No JSON files found in {source_dir}")
        return source_data

    for json_file in json_files:
        with open(json_file, "r") as f:
            contents = f.read()

        nuke_files = json.loads(contents)
        for file_path, script_contents in nuke_files.items():
            new_path = os.path.join(source_dir, os.path.basename(file_path))
            with open(new_path, "w", encoding="utf-8") as f:
                f.write(script_contents)
            log.info(f"Restored: {new_path}")


def save_model_checkpoint(model: "NukeGATPredictor", model_name: str):
    import torch

    os.makedirs(DirectoryConfig.MODEL_PATH, exist_ok=True)

    model_path = os.path.join(DirectoryConfig.MODEL_PATH, f"{model_name}_model.pt")
    checkpoint = {
        "num_features": model.num_features,
        "state_dict": model.state_dict(),
        "hidden_channels": model.hidden_channels,
        "num_layers": model.num_layers,
        "num_heads": model.heads,
        "dropout": model.dropout,
    }

    torch.save(checkpoint, model_path)
    log.info(f"Model saved to {DirectoryConfig.MODEL_PATH}")


def load_model_checkpoint(model_name: str, device: Optional[str] = "cuda"):
    import torch

    model_checkpoint_path = os.path.join(
        DirectoryConfig.MODEL_PATH, f"{model_name}_model.pt"
    )

    # Load checkpoint
    checkpoint = torch.load(model_checkpoint_path, map_location=device)

    # Load vocabulary
    with open(DirectoryConfig.VOCAB_PATH, "r") as f:
        vocab = json.load(f)

    # Initialize model
    from gat import NukeGATPredictor

    model = NukeGATPredictor(
        num_features=4,
        num_classes=vocab["num_node_types"],
        hidden_channels=checkpoint["hidden_channels"],
        num_layers=checkpoint["num_layers"],
        heads=checkpoint["num_heads"],
        dropout=checkpoint["dropout"],
    ).to(device)

    # Load model state
    model.load_state_dict(check_state_dict(checkpoint["state_dict"]))
    log.info("Successfully loaded state dictionary.")

    # Load scaler if it was saved
    scaler = None
    if "scaler_state" in checkpoint:
        from torch.amp import GradScaler

        scaler = GradScaler(device)
        scaler.load_state_dict(checkpoint["scaler_state"])

    return model, scaler, vocab


def check_state_dict(state_dict: Dict[str, Any], unwanted_prefixes: List[str] = None):
    if unwanted_prefixes is None:
        unwanted_prefixes = ["_orig_mod.", "module."]

    for k, v in list(state_dict.items()):
        for unwanted_prefix in unwanted_prefixes:
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    return state_dict
