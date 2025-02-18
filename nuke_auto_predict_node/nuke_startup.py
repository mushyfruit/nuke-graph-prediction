import logging

import nuke
import nukescripts

from .launcher import launch_inference_service
from .ui import prediction_panel
from .recommendation import perform_recommendation, load_model_vocab


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_model_vocabulary = None


def on_startup():
    """Initialize ML service on Nuke startup"""
    try:
        load_model_vocab()

        launch_inference_service()

        register_prediction_panel()

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


def register_prediction_panel():
    pane_menu = nuke.menu("Pane")
    pane_menu.addCommand(
        "Node Recommendation✨", prediction_panel.show_prediction_panel
    )
    nukescripts.registerPanel(
        "mushyfruit.node_recommendation", prediction_panel.show_prediction_panel
    )
