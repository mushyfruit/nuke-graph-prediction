from functools import lru_cache

import nuke
import nukescripts

BUILT_IN_TABS = [
    "Properties.1",
    "DAG.1",
    "DopeSheet.1",
    "Viewer.1",
    "Toolbar.1",
]


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
    from ...api.recommendation import get_prediction_manager

    manager = get_prediction_manager()
    return PredictionWidget(manager)


class PredictionPanel(nukescripts.panels.PythonPanel):
    def __init__(self):
        nukescripts.panels.PythonPanel.__init__(
            self, "NodeScout", "mushyfruit.node_scout"
        )

        self.prediction_widget = nuke.PyCustom_Knob(
            "prediction_widget",
            "",
            ("node_scout.ui.nuke.prediction_panel.create_prediction_widget_instance()"),
        )
        self.prediction_widget.setFlag(nuke.STARTLINE)
        self.addKnob(self.prediction_widget)
