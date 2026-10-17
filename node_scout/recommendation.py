from typing import Optional

from .api.recommendation import get_prediction_manager


def perform_recommendation(node_name: Optional[str] = None) -> None:
    get_prediction_manager().perform_recommendation(node_name=node_name)
