import os


class NukeScript:
    ROOT = "root"
    NODES = "nodes"


PLUGIN_ROOT = os.path.dirname(os.path.dirname(__file__))

VOCAB = "vocab.json"
MODEL_NAME = "nuke_predictor_gat"

MODEL_PATH = os.path.join(PLUGIN_ROOT, "checkpoints")
MODEL_DATA_FOLDER = os.path.join(PLUGIN_ROOT, "tmp", "nuke_graphs")
DATA_CACHE_PATH = os.path.join(MODEL_PATH, "data_cache")
VOCAB_PATH = os.path.join(DATA_CACHE_PATH, VOCAB)
