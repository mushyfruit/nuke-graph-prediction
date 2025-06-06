from functools import lru_cache

from typing import Any, Dict

from ...logging_config import get_logger

import nuke
import nukescripts

BUILT_IN_TABS = [
    "Properties.1",
    "DAG.1",
    "DopeSheet.1",
    "Viewer.1",
    "Toolbar.1",
]


log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_panel_instance():
    return PredictionPanel()


def show_prediction_panel():
    panel = get_panel_instance()

    existing_pane = None
    for tab_name in BUILT_IN_TABS:
        existing_pane = nuke.getPaneFor(tab_name)
        if existing_pane:
            break

    return panel.addToPane(pane=existing_pane)


def create_prediction_widget_instance():
    from .prediction_widget import PredictionWidget
    from ...api.request_handler import RequestHandler
    from ...api.recommendation import get_prediction_manager

    try:
        manager = get_prediction_manager()
        handler = RequestHandler()
        return PredictionWidget(handler, manager)
    except Exception as e:
        raise RuntimeError(f"Failed to create prediction widget: {e}")


class PredictionPanel(nukescripts.panels.PythonPanel):
    def __init__(self):
        nukescripts.panels.PythonPanel.__init__(
            self, "NodeScout", "mushyfruit.node_scout"
        )

        self._init_layout()

    def _init_layout(self):
        self.prediction_widget = nuke.PyCustom_Knob(
            "prediction_widget",
            "",
            ("node_scout.ui.nuke.prediction_panel.create_prediction_widget_instance()"),
        )
        self.prediction_widget.setFlag(nuke.STARTLINE)
        self.addKnob(self.prediction_widget)

    def update_training_page_label(self, label: str):
        prediction_widget = self.prediction_widget.getObject()
        if prediction_widget:
            prediction_widget.update_prediction_page_status(label)

    def update_prediction_state(self, prediction: Dict[str, Any]):
        prediction_list = prediction.get("prediction", None)
        if prediction_list is None:
            log.error(
                f"Unable to retrieve valid prediction from response: {prediction}"
            )
            return

        start_node = prediction["start_node"]
        prediction_widget = self.prediction_widget.getObject()

        if prediction_widget:
            prediction_widget.update_selected_node(start_node)
            prediction_widget.update_prediction(prediction_list)
            prediction_widget.update_prediction_page_status("")
