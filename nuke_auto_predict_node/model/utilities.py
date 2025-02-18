import os
import json
from dotenv import load_dotenv
from subprocess import Popen, PIPE
from tqdm import tqdm

from .constants import VOCAB_PATH, MODEL_PATH


def json_back_to_nk():
    import glob

    current_dir = os.path.dirname(os.path.dirname(__file__))
    source_dir = os.path.join(current_dir, "test_scripts")

    source_data = {}

    # Create the pattern to match all JSON files
    json_pattern = os.path.join(source_dir, "*.json")

    json_files = glob.glob(json_pattern)
    if not json_files:
        print(f"No JSON files found in {source_dir}")
        return source_data

    for json_file in json_files:
        with open(json_file, "r") as f:
            contents = f.read()

        nuke_files = json.loads(contents)
        for file_path, script_contents in nuke_files.items():
            new_path = os.path.join(source_dir, os.path.basename(file_path))
            with open(new_path, "w", encoding="utf-8") as f:
                f.write(script_contents)
            print(f"Restored: {new_path}")


def get_remote_info():
    load_dotenv()

    remote_server = os.getenv("REMOTE_SERVER")
    if remote_server is None:
        raise RuntimeError("No REMOTE_SERVER environment variable set!")

    remote_dir = os.getenv("REMOTE_DIR")
    if remote_dir is None:
        raise RuntimeError("No REMOTE_DIR environment variable set!")

    return remote_server, remote_dir


def list_remote_files(ext=".json"):
    remote_server, remote_dir = get_remote_info()

    ssh_command = ["ssh", remote_server, f"ls {remote_dir}"]
    proc = Popen(ssh_command, stdout=PIPE, universal_newlines=True)
    stdout = proc.stdout.read()
    return [f for f in stdout.strip().split("\n") if f.endswith(ext)]


def download_remote_files(output_dir):
    # Ensure the nuke_graphs directory exists.
    os.makedirs(output_dir, exist_ok=True)
    remote_files = list_remote_files()

    # Download each file if it doesn't exist locally
    with tqdm(remote_files, desc="Downloading files") as pbar:
        for filename in pbar:
            local_path = os.path.join(output_dir, filename)
            if not os.path.exists(local_path):
                pbar.set_description(f"Downloading {filename}")
                download_file(filename, local_path)


def download_file(filename, local_path):
    remote_server, remote_dir = get_remote_info()
    scp_command = ["scp", f"{remote_server}:{remote_dir}/{filename}", local_path]
    proc = Popen(scp_command)
    proc.wait()


def save_model_checkpoint(model, model_name):
    import torch

    os.makedirs(MODEL_PATH, exist_ok=True)

    model_path = os.path.join(MODEL_PATH, f"{model_name}_model.pt")
    checkpoint = {
        "num_features": model.num_features,
        "state_dict": model.state_dict(),
        "hidden_channels": model.hidden_channels,
        "num_layers": model.num_layers,
        "num_heads": model.heads,
        "dropout": model.dropout,
    }

    torch.save(checkpoint, model_path)
    print(f"Model saved to {MODEL_PATH}")


def load_model_checkpoint(model_name: str, device="cuda"):
    import torch

    model_checkpoint_path = os.path.join(MODEL_PATH, f"{model_name}_model.pt")

    # Load checkpoint
    checkpoint = torch.load(model_checkpoint_path, map_location=device)

    # Load vocabulary
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)

    # Initialize model
    from model import NukeGATPredictor

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
    print("Successfully loaded state dictionary.")

    # Load scaler if it was saved
    scaler = None
    if "scaler_state" in checkpoint:
        from torch.amp import GradScaler

        scaler = GradScaler(device)
        scaler.load_state_dict(checkpoint["scaler_state"])

    return model, scaler, vocab


def check_state_dict(state_dict, unwanted_prefixes=None):
    if unwanted_prefixes is None:
        unwanted_prefixes = ["_orig_mod.", "module."]

    for k, v in list(state_dict.items()):
        for unwanted_prefix in unwanted_prefixes:
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    return state_dict
