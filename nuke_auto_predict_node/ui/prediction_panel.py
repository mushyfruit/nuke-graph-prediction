import nuke
import nukescripts

import logging

from typing import List, Tuple, Dict

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BUILT_IN_TABS = [
    "Properties.1",
    "DAG.1",
    "DopeSheet.1",
    "Viewer.1",
    "Toolbar.1",
]

_g_prediction_panel = None


def get_panel_instance():
    global _g_prediction_panel
    if not _g_prediction_panel:
        _g_prediction_panel = PredictionPanel()
    return _g_prediction_panel


def show_prediction_panel():
    panel = get_panel_instance()

    existing_pane = None
    for tab_name in BUILT_IN_TABS:
        existing_pane = nuke.getPaneFor(tab_name)
        if existing_pane:
            break

    return panel.addToPane(pane=existing_pane)


def create_prediction_widget_instance(request_handler=None):
    from .prediction_widget import PredictionWidget
    from ..request_handler import get_request_handler

    try:
        handler = request_handler or get_request_handler()
        return PredictionWidget(handler)
    except Exception as e:
        raise RuntimeError(f"Failed to craete prediction widget: {e}")


class PredictionPanel(nukescripts.panels.PythonPanel):
    def __init__(self):
        nukescripts.panels.PythonPanel.__init__(
            self, "NodePrediction", "mushyfruit.node_recommendation"
        )

        self._init_layout()

    def _init_layout(self):
        self.prediction_widget = nuke.PyCustom_Knob(
            "prediction_widget",
            "",
            (
                "nuke_auto_predict_node.ui.prediction_panel.create_prediction_widget_instance()"
            ),
        )
        self.prediction_widget.setFlag(nuke.STARTLINE)
        self.addKnob(self.prediction_widget)

    def update_prediction_state(
        self, selected_node: str, prediction: Dict[str, List[Tuple[str, float]]]
    ):
        prediction_widget = self.prediction_widget.getObject()
        if prediction_widget:
            prediction_widget.update_selected_node(selected_node)
            prediction_widget.update_prediction(prediction)
