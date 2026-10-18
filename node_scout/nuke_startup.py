import nuke
import nukescripts

from .api.recommendation import get_prediction_manager
from .server.launcher import get_inference_launcher
from .ui.nuke import prediction_panel


def on_startup():
    try:
        get_inference_launcher().start_service()
    except Exception as e:
        nuke.message(f"Failed to start ML service: {e}")
        return

    if not nuke.GUI:
        return

    try:
        pane_menu = nuke.menu("Pane")
        pane_menu.addCommand("NodeScout", prediction_panel.show_prediction_panel)
        nukescripts.registerPanel(
            "mushyfruit.node_scout", prediction_panel.show_prediction_panel
        )

        nuke.menu("Nuke").addCommand(
            "Edit/Node/Auto-Predict",
            lambda: get_prediction_manager().perform_recommendation(),
            "ctrl+shift+t",
        )

    except Exception as e:
        nuke.message(f"Failed to register NodeScout menus: {e}")
