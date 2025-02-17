import os
import json
import logging

import nuke

from .launcher import launch_inference_service
from .request_handler import get_request_handler
from .ui import prediction_panel


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_model_vocabulary = None


def load_model_vocab():
    global _model_vocabulary
    from . import model_cnst

    vocab_path = os.path.join(model_cnst.DATA_CACHE_PATH, model_cnst.VOCAB)
    if not os.path.exists(vocab_path):
        log.error("No vocab file found at {}".format(vocab_path))
        return

    with open(vocab_path, "r") as f:
        _model_vocabulary = json.load(f).get("node_type_to_idx", {})


def nuke_startup():
    """Initialize ML service on Nuke startup"""
    try:
        load_model_vocab()

        launch_inference_service()

        prediction_panel.register_prediction_panel()
        nuke.addKnobChanged(predict_selected)

        nuke.menu("Nuke").addCommand(
            "Recommendation/PerformTest",
            "nuke_auto_predict_node.recommendation.perform_recommendation()",
            "ctrl+shift+t",
        )

    except Exception as e:
        nuke.message(f"Failed to start ML service: {str(e)}")


def predict_selected():
    if nuke.thisKnob().name() != "selected":
        return
    perform_recommendation()


def is_node_disabled(node):
    disable_knob = node.knob("disable")
    if not disable_knob:
        return False

    return disable_knob.value()


def traverse_upstream(node, target_nodes=None, visited=None, length=0, max_length=15):
    if target_nodes is None:
        target_nodes = {}

    if visited is None:
        visited = set()

    if length >= max_length:
        return target_nodes, visited

    node_full_name = node.fullName()
    if node_full_name in visited:
        return target_nodes, visited

    if not is_node_disabled(node):
        # Ensure our model has seen the node.
        if _model_vocabulary is None:
            load_model_vocab()

        if node.Class() in _model_vocabulary:
            target_nodes[node_full_name] = serialize_node(node)

    dependency_flag = nuke.INPUTS | nuke.HIDDEN_INPUTS
    ancestors = node.dependencies(dependency_flag)

    for ancestor in ancestors:
        if ancestor.Class() in {"Group", "Gizmo"} or "gizmo_file" in ancestor.knobs():
            traverse_upstream(
                ancestor, target_nodes=target_nodes, visited=visited, length=length + 1
            )

        traverse_upstream(
            ancestor, target_nodes=target_nodes, visited=visited, length=length + 1
        )

    return target_nodes


def serialize_node(node, include_parameters=False):
    dependency_flag = nuke.INPUTS | nuke.HIDDEN_INPUTS
    ancestors = node.dependencies(dependency_flag)

    json_node = {
        "name": node.name(),
        "node_type": node.Class(),
        "inputs": node.inputs(),
        "input_connections": [ancestor.name() for ancestor in ancestors],
    }

    if include_parameters:
        json_node["parameters"] = get_all_parameters(node)

    return json_node


def get_all_parameters(node):
    blacklist_knobs = {"xpos", "ypos"}

    parameter_dict = {}
    for knob in node.knobs():
        if knob in blacklist_knobs:
            continue

        k = node[knob]
        knob_value = k.value()

        try:
            if k.defaultValue() == knob_value:
                continue
        # Not all knobs define a 'defaultValue()'
        except Exception as e:
            pass

        if not isinstance(knob_value, (str, float, int)):
            continue

        if knob_value == "":
            continue

        parameter_dict[knob] = node[knob].value()

    return parameter_dict


def serialize_upstream():
    selected_node = nuke.selectedNode()
    if not selected_node:
        nuke.message("Please select a node!")
        return

    serialized_nodes = traverse_upstream(selected_node)
    return {
        "script_name": os.path.basename(nuke.scriptName()),
        "root": {"name": "root", "parent": None, "nodes": serialized_nodes},
        "start_node": selected_node.name(),
    }


def perform_recommendation():
    try:
        serialized_graph_data = serialize_upstream()
    except (ValueError, RuntimeError):
        return

    log.info(serialized_graph_data)

    if len(serialized_graph_data["root"]["nodes"]) <= 3:
        return

    handler = get_request_handler()
    panel_instance = prediction_panel.get_panel_instance()

    log.info("Making prediction POST request...")
    prediction = handler.post("predict", serialized_graph_data)

    log.info(f"Prediction: {prediction}")
    if panel_instance:
        panel_instance.show_prediction(prediction)
