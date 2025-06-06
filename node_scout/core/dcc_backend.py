from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable


class DCCBackend(ABC):
    @abstractmethod
    def get_node_name(self, node: Any) -> str:
        pass

    @abstractmethod
    def get_target_node(self, node_name: Optional[str] = None) -> Optional[Any]:
        pass

    @abstractmethod
    def serialize_upstream_nodes(self, start_node: Any) -> Dict:
        pass

    @abstractmethod
    def serialize_node(self, node: Any) -> Dict:
        pass

    @abstractmethod
    def should_perform_callback(self) -> bool:
        pass

    @abstractmethod
    def enable_callback(self, callback_fn: Callable[[], None]) -> None:
        pass

    @abstractmethod
    def disable_callback(self, callback_fn: Callable[[], None]) -> None:
        pass

    @abstractmethod
    def validate_node_for_inference(self, node: Any) -> bool:
        pass

    @abstractmethod
    def show_message(self, text: str) -> None:
        pass
