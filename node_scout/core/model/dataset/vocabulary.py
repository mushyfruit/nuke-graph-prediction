import os
import json
import logging
from typing import Dict, Optional

from torch_geometric.data import Data

log = logging.getLogger(__name__)


class Vocabulary:
    def __init__(self, vocab_file: Optional[str] = None):
        self._type_to_idx: Dict[str, int] = {}
        self._idx_to_type: Dict[int, str] = {}

        if vocab_file:
            self.load(vocab_file)

    def get_idx(self, node_type: str) -> int:
        return self.add(node_type)

    def get_type(self, idx: int) -> str:
        return self._idx_to_type[idx]

    def add(self, node_type: str) -> int:
        """Adds a new node type to the vocabulary if it doesn't exist."""
        if node_type not in self._type_to_idx:
            idx = len(self._type_to_idx)
            self._type_to_idx[node_type] = idx
            self._idx_to_type[idx] = node_type
            return idx
        return self._type_to_idx[node_type]

    def load(self, file_path: str) -> None:
        """Loads the vocabulary from a JSON file."""
        if not os.path.exists(file_path):
            log.warning(f"Vocabulary file {file_path} not found.")
            return

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            self._type_to_idx = data["node_type_to_idx"]
            self._idx_to_type = {idx: type_ for type_, idx in self._type_to_idx.items()}
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid vocabulary file format: {e}")

    def save(self, file_path: str) -> None:
        """Saves the vocabulary to a JSON file."""
        data = {"node_type_to_idx": self._type_to_idx, "num_node_types": len(self)}

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def __contains__(self, node_type: str) -> bool:
        return node_type in self._type_to_idx

    def __len__(self) -> int:
        return len(self._type_to_idx)
