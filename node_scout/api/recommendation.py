import logging
from functools import lru_cache

from .request_handler import RequestHandler

from ..core.model.utilities import check_for_model_on_disk

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_prediction_manager():
    return PredictionManager()


class PredictionManager:
    def __init__(self):
        from ..core.nuke.nuke_backend import NukeBackend

        self._backend = NukeBackend()
        self._request_handler = RequestHandler()
        self._min_node_requirement = 3
        self._model_exists = check_for_model_on_disk()

    def toggle_callback(self, enable_callback: bool):
        if enable_callback:
            self._backend.enable_callback(self.predict_selected)
        else:
            self._backend.disable_callback(self.predict_selected)

    def predict_selected(self):
        if not self._backend.should_perform_callback():
            return

        self.perform_recommendation()

    def perform_recommendation(self, node=None) -> None:
        widget = self._get_widget()
        widget.clear_prediction()
        try:
            target_node = self._backend.get_target_node(node)
            if not target_node:
                widget.update_prediction_page_status("Please select a node!")
                return

            if not self._backend.validate_node_for_inference(target_node):
                widget.update_prediction_page_status(
                    f"No prediction available for {target_node.fullName()}!"
                )
                return

            if not self._model_exists:
                widget.update_prediction_page_status("Please train a local model first!")
                return

            graph_data = self._backend.serialize_upstream_nodes(target_node)
            if graph_data["start_node"] is None:
                widget.update_prediction_page_status(
                    f"No prediction available for {target_node.fullName()}!"
                )
                return
            if len(graph_data["root"]["nodes"]) <= self._min_node_requirement:
                widget.update_prediction_page_status(
                    f"Not enough upstream nodes to predict for {target_node.name()}!"
                )
                return

            widget.request_prediction(graph_data, target_node)

        except Exception as e:
            log.error(f"Error during recommendation: {e}")
            widget.clear_prediction()
            widget.update_prediction_page_status(str(e))

    def refresh_manager(self):
        model_exists = check_for_model_on_disk()
        if model_exists and not self._model_exists:
            self._get_widget().update_prediction_page_status(
                "Found valid PyTorch model."
            )

        self._model_exists = model_exists

    def _get_widget(self):
        from ..ui.nuke.prediction_panel import get_panel_instance

        panel = get_panel_instance()
        widget = panel.prediction_widget.getObject()
        if widget is None:
            panel.show()
            widget = panel.prediction_widget.getObject()
        return widget
