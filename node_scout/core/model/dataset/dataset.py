import os
import json
import pickle
import logging
from tqdm import tqdm

from .vocabulary import Vocabulary
from .deserialize import create_graph_data
from ..constants import NukeScript, DirectoryConfig, VOCAB

import torch
from torch_geometric.data import Data, Dataset

from typing import Dict, Optional, Any

log = logging.getLogger(__name__)


class GraphDataset(Dataset):
    def __init__(
        self,
        root_dir,
        force_rebuild: bool = False,
    ):
        super().__init__(root=root_dir)

        self.root_dir = root_dir
        self.file_paths = []

        self.processed_files = []
        self.examples = []

        self.processed_graphs_file = os.path.join(
            DirectoryConfig.DATA_CACHE_PATH, "process_graphs.pt"
        )
        self.metadata_file = os.path.join(
            DirectoryConfig.DATA_CACHE_PATH, "graph_metadata.json"
        )

        # Load the Nuke node type vocabulary.
        self.vocabulary_path = os.path.join(DirectoryConfig.DATA_CACHE_PATH, VOCAB)
        self.vocab = Vocabulary(self.vocabulary_path)

        if not force_rebuild:
            self._load_cache()

        self.process_all_graphs_in_dir(self.root_dir)

    def _load_cache(self) -> None:
        # Check if the files have already been processed.
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                self.processed_files = json.load(f).get("processed_files", [])

        # Load processed graph data.
        if os.path.exists(self.processed_graphs_file):
            saved_data = torch.load(self.processed_graphs_file, weights_only=False)
            self.examples = saved_data["examples"]

    def process_all_graphs_in_dir(self, target_dir: str) -> None:
        for file in os.listdir(target_dir):
            if file.endswith(".json"):
                self.file_paths.append(os.path.join(target_dir, file))

        for file_path in tqdm(self.file_paths, desc="Processing all graphs"):
            # We've already processed this file.
            if file_path in self.processed_files:
                continue

            with open(file_path, "r") as f:
                data = json.load(f)

            graph_examples = self.generate_graph_training_examples(data)
            self.examples.extend(graph_examples)
            self.processed_files.append(file_path)

        self.save_graph_state()

        log.info(f"Processed {len(self.examples)} total examples")

    def save_graph_state(self) -> None:
        # Ensure the metadata parent dir exists.
        os.makedirs(DirectoryConfig.DATA_CACHE_PATH, exist_ok=True)

        torch.save(
            {
                "examples": self.examples,
            },
            self.processed_graphs_file,
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
        )
        self._save_metadata()
        self.vocab.save(self.vocabulary_path)

    def _save_metadata(self) -> None:
        with open(self.metadata_file, "w") as f:
            json.dump(
                {
                    "processed_files": self.processed_files,
                },
                f,
                indent=2,
            )

    def generate_graph_training_examples(
        self,
        data: Dict[str, Any],
        min_context: Optional[int] = 5,
        max_context: Optional[int] = 50,
        stride: Optional[int] = 3,
    ):
        root_group = data[NukeScript.ROOT]
        nodes = root_group[NukeScript.NODES]

        graph_node_data = list(nodes.values())
        examples = []

        for i in reversed(range(min_context, len(graph_node_data), stride)):
            # This is the prediction node.
            target_node_data = graph_node_data[i]
            graph_data = create_graph_data(
                data,
                target_node_data,
                self.vocab,
                update_vocab=True,
                min_upstream_nodes=min_context,
                max_upstream_nodes=max_context,
                filter_graphs=True,
            )
            if not graph_data:
                continue

            # Ensure we include graph-level ground-truth label.
            target = torch.tensor(
                self.vocab.get_idx(target_node_data["node_type"]),
                dtype=torch.long,
            )
            graph_data.y = target
            graph_data.validate(raise_on_error=True)

            # All invalid graphs should have already been filtered.
            if graph_data.num_nodes == 0:
                continue

            examples.append(graph_data)

        return examples

    def len(self) -> int:
        return len(self.file_paths)

    def get(self, idx: int) -> Data:
        return self.examples[idx]
