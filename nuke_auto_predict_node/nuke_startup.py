import nuke
import nukescripts

from .server.launcher import launch_inference_service
from .ui import prediction_panel

_model_vocabulary = None


def on_startup():
    """Initialize ML service on Nuke startup"""
    try:
        launch_inference_service()
        register_prediction_panel()

        nuke.menu("Nuke").addCommand(
            "Recommendation/PerformTest",
            "nuke_auto_predict_node.recommendation.perform_recommendation()",
            "ctrl+shift+t",
        )

    except Exception as e:
        nuke.message(f"Failed to start ML service: {str(e)}")


def register_prediction_panel():
    pane_menu = nuke.menu("Pane")
    pane_menu.addCommand(
        "Node Recommendation✨", prediction_panel.show_prediction_panel
    )
    nukescripts.registerPanel(
        "mushyfruit.node_recommendation", prediction_panel.show_prediction_panel
    )
