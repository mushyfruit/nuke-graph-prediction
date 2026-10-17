import nuke
import nukescripts

from . import recommendation
from .server.launcher import launch_inference_service
from .ui.nuke import prediction_panel


def on_startup():
    try:
        launch_inference_service()

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
