import os
import json
import hashlib
import logging
from tqdm import tqdm

from .vocabulary import Vocabulary
from .deserialize import create_graph_data
from ..constants import DirectoryConfig

import torch
from typing import Dict, Any

log = logging.getLogger(__name__)


class GraphDataset:
    def __init__(self, root_dir):
        self.file_paths = [
            os.path.join(root_dir, file)
            for file in os.listdir(root_dir)
            if file.endswith(".json")
        ]
        file_hashes = {}
        for file_path in self.file_paths:
            with open(file_path, "rb") as f:
                file_hashes[file_path] = hashlib.sha256(f.read()).hexdigest()

        cache_path = os.path.join(DirectoryConfig.DATA_CACHE_PATH, "process_graphs.pt")
        self.vocab = Vocabulary(DirectoryConfig.VOCAB_PATH)

        if os.path.exists(cache_path):
            saved_data = torch.load(cache_path, weights_only=False)
            if saved_data.get("files") == file_hashes and saved_data.get("vocab") == [
                self.vocab.get_type(i) for i in range(len(self.vocab))
            ]:
                self.examples = saved_data["examples"]
                return

        self.examples = []
        for file_path in tqdm(self.file_paths, desc="Processing all graphs"):
            with open(file_path, "r") as f:
                data = json.load(f)

            self.examples.extend(self.generate_graph_training_examples(data))

        self.vocab.save(DirectoryConfig.VOCAB_PATH)
        torch.save(
            {
                "examples": self.examples,
                "files": file_hashes,
                "vocab": [self.vocab.get_type(i) for i in range(len(self.vocab))],
            },
            cache_path,
        )
        log.info(f"Processed {len(self.examples)} total examples")

    def generate_graph_training_examples(self, data: Dict[str, Any]):
        nodes = data["root"]["nodes"]

        graph_node_data = list(nodes.values())
        examples = []

        for i in reversed(range(5, len(graph_node_data), 3)):
            target_node_data = graph_node_data[i]
            graph_data = create_graph_data(
                data,
                target_node_data,
                self.vocab,
                update_vocab=True,
                min_upstream_nodes=5,
            )
            if graph_data is None:
                continue

            graph_data.y = torch.tensor(
                self.vocab.add(target_node_data["node_type"]),
                dtype=torch.long,
            )
            graph_data.validate(raise_on_error=True)
            examples.append(graph_data)

        return examples

    @property
    def num_features(self) -> int:
        return self.examples[0].num_node_features if self.examples else 0
