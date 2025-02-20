import nuke
import nukescripts

from .server.launcher import launch_inference_service
from .ui import prediction_panel
from .api.recommendation import get_prediction_manager

_model_vocabulary = None


def on_startup():
    """Initialize ML service on Nuke startup"""
    global _prediction_manager
    try:
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
    knob = nuke.thisKnob()
    if knob is None or knob.name() != "selected":
        return

    # Return early when deselecting nodes.
    if not knob.value():
        return

    prediction_manager = get_prediction_manager()
    prediction_manager.perform_recommendation()



def register_prediction_panel():
    pane_menu = nuke.menu("Pane")
    pane_menu.addCommand(
        "Node Recommendation✨", prediction_panel.show_prediction_panel
    )
    nukescripts.registerPanel(
        "mushyfruit.node_recommendation", prediction_panel.show_prediction_panel
    )
