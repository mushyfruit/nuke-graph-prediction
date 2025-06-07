import nuke
import nukescripts

from .server.launcher import launch_inference_service
from .ui.nuke import prediction_panel

_model_vocabulary = None


def on_startup():
    """Initialize ML service on Nuke startup"""
    try:
        # Start inference service in a separate thread.
        launch_inference_service()

        # Register the python panel.
        register_prediction_panel()

        nuke.menu("Nuke").addCommand(
            "Edit/Node/Auto-Predict",
            "node_scout.recommendation.perform_recommendation()",
            "ctrl+shift+t",
        )

    except Exception as e:
        nuke.message(f"Failed to start ML service: {str(e)}")


def register_prediction_panel():
    pane_menu = nuke.menu("Pane")
    pane_menu.addCommand("NodeScout", prediction_panel.show_prediction_panel)
    nukescripts.registerPanel(
        "mushyfruit.node_scout", prediction_panel.show_prediction_panel
    )
