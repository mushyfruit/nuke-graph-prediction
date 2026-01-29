"""Convenience wrapper for Nuke menu commands.

Nuke's `addCommand` API often takes a string to execute. Keeping that string
stable and short makes installation/configuration easier.

This module intentionally re-exports the recommendation entry point.
"""

from typing import Optional

from .api.recommendation import get_prediction_manager


def perform_recommendation(node_name: Optional[str] = None) -> None:
    """Perform a next-node recommendation.

    If `node_name` is not provided, the active/selected node is used.
    """

    get_prediction_manager().perform_recommendation(node_name=node_name)
