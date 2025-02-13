import os

class NukeScript:
    ROOT = "root"
    NODES = "nodes"

INVALID_NODE_CLASSES = {"Read", "Input", "Constant", "CheckerBoard2", "ReadGeo2"}

PLUGIN_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_NAME = "nuke_predictor_gat"
MODEL_PATH = os.path.join(PLUGIN_ROOT, "checkpoints")
VOCAB = "vocab.json"
MODEL_DATA_FOLDER = os.path.join(PLUGIN_ROOT, "tmp", "nuke_graphs")