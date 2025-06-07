import os
import queue
import logging
import threading
import traceback
from dataclasses import asdict
from typing import Optional, List, Tuple, Any

import torch
from torch_geometric.data import Data

from .main import train_model_gat, TrainingConfig
from .queue import StatusQueue
from .dataset.vocabulary import Vocabulary
from .dataset.dataset import GraphDataset
from .gat import NukeGATPredictor
from .constants import (
    MODEL_NAME,
    DirectoryConfig,
    TrainingPhase,
    TrainingStatus,
)
from .utilities import check_state_dict, save_model_checkpoint, check_for_model_on_disk
from ..nuke.parser import NukeScriptParser
from ..nuke.serialization import NukeGraphSerializer

log = logging.getLogger(__name__)


class GNNModelController:
    def __init__(self, model_class, model_config, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._vocab = None

        self._model_class = model_class
        self._model_config = model_config
        self._model_name = model_name

        # Thread-safe queue for communicating status updates through endpoint.
        self.status_queue = StatusQueue(maxsize=100)

        self.training_thread = None
        self._model_lock = threading.RLock()

        # Flag to track model training status.
        self._is_training = threading.Event()

        # Load the model and vocabulary.
        self.load()

    def get_vocabulary(self):
        return self._vocab

    def load(self) -> bool:
        """Instantiate the target GNN model and populate the model's stored vocabulary."""
        # Retrieve the checkpoint's path.
        if not check_for_model_on_disk(self._model_name):
            log.info(f"Model {self._model_name} not found on disk. Skipping load.")
            return False

        if not os.path.exists(DirectoryConfig.VOCAB_PATH):
            raise FileNotFoundError(f"Vocab {DirectoryConfig.VOCAB_PATH} not found")

        # Load the model checkpoint.
        model_checkpoint_path = os.path.join(
            DirectoryConfig.MODEL_PATH, f"{self._model_name}_model.pt"
        )
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)

        # Populate the model's stored vocabulary.
        self._vocab = Vocabulary(DirectoryConfig.VOCAB_PATH)

        # Instantiate the GAT model.
        self._model = self._model_class.from_checkpoint(
            checkpoint, num_classes=len(self._vocab)
        )

        # Restore the state dictionary.
        self._model.load_state_dict(check_state_dict(checkpoint["state_dict"]))

        # Set model into inference mode.
        self._model.eval()

        return True

    def get_training_model(self, dataset: GraphDataset, fine_tune: bool = False) -> Any:
        # TODO: Specify model settings via new python panel page.
        config = asdict(TrainingConfig())

        training_model = self._model_class(
            num_features=4, num_classes=len(dataset.vocab), **config
        )

        if fine_tune:
            training_model.load_state_dict(check_state_dict(self._model.state_dict()))

        return training_model

    def start_training_pipeline(
        self, file_paths: List[str], memory_allocation: float, enable_fine_tuning: bool
    ) -> TrainingStatus:
        if self._is_training.is_set():
            return TrainingStatus(
                phase=TrainingPhase.SERIALIZING, label="Pipeline already in progress"
            )

        log.info("Starting the training pipeline!")
        while not self.status_queue.empty():
            self.status_queue.get_nowait()

        self._is_training.set()

        log.info("Starting the training thread...")
        self.training_thread = threading.Thread(
            target=self._training_thread_target,
            args=(file_paths, memory_allocation, enable_fine_tuning),
        )
        self.training_thread.start()

        # Start the process and immediately return.
        return TrainingStatus(
            phase=TrainingPhase.SERIALIZING, label="Pipeline started successfully..."
        )

    def _training_thread_target(
        self, file_paths: List[str], memory_allocation: float, enable_fine_tuning: bool
    ) -> None:
        try:
            # Start serialization.
            output_dir = self.parse_and_serialize_scripts(file_paths)
            self.status_queue.safe_put(
                TrainingStatus(
                    phase=TrainingPhase.SERIALIZATION_COMPLETE,
                    label="Script serialization complete!",
                    progress=1.0,
                )
            )

            # Begin the training phase.
            log.info("Force rebuild activated...")
            dataset = GraphDataset(output_dir, force_rebuild=True)

            log.info("processing")
            dataset.process_all_graphs_in_dir(output_dir)

            self.status_queue.safe_put(
                TrainingStatus(
                    phase=TrainingPhase.TRAINING,
                    label="Starting model training...",
                    progress=0.0,
                )
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            log.info(
                f"Starting model training: "
                f"Memory allocation: {memory_allocation}, "
                f"fine-tuning: {enable_fine_tuning}"
            )

            training_model = self.get_training_model(
                dataset, fine_tune=enable_fine_tuning
            )
            trained_model = train_model_gat(
                dataset,
                training_model,
                memory_fraction=memory_allocation,
                status_queue=self.status_queue,
            )
            save_model_checkpoint(trained_model, MODEL_NAME)

            # Swap in the trained model.
            # Allow for inference with old model during training.
            with self._model_lock:
                self._model = trained_model

            self.status_queue.safe_put(
                TrainingStatus(
                    phase=TrainingPhase.COMPLETE,
                    label="Finished training the model!",
                    progress=1.0,
                )
            )

        except Exception as e:
            log.error(traceback.format_exc())
            raise
        finally:
            self._is_training.clear()

    def get_training_status(self) -> TrainingStatus:
        """Get current training status"""
        if self.status_queue.empty():
            if not self._is_training.is_set():
                return TrainingStatus(
                    phase=TrainingPhase.IDLE, label="Waiting for execution..."
                )
            return TrainingStatus(phase=TrainingPhase.NO_CHANGE)

        return self.status_queue.get_latest()

    def predict(self, pyg_graph_data: Data) -> List[Tuple[int, float]]:
        if self._model is None:
            if not self.load():
                raise RuntimeError("Model is not loaded!")

        test_data = pyg_graph_data.to(self.device)

        # Disable gradient computation.
        with torch.no_grad():
            with self._model_lock:
                # Get predictions over all possible node types.
                # Ensure thread-safe inference.
                predictions = self._model(test_data)

            # Convert logits to probabilities.
            probabilities = torch.nn.functional.softmax(predictions, dim=1)

            # Select top X from probabilities. [1, 10]
            top_probs, top_indices = torch.topk(probabilities, k=10)

            node_predictions = []
            for idx, prob in zip(top_indices[0], top_probs[0]):
                # Convert the idx to Nuke node type name.
                node_type = self._vocab.get_type(idx.item())
                probability = prob.item()
                node_predictions.append((node_type, probability))

            return node_predictions

    def parse_and_serialize_scripts(
        self, script_paths: List[str], output_dir: Optional[str] = None
    ) -> str:
        if output_dir is None:
            output_dir = DirectoryConfig.MODEL_DATA_FOLDER

        os.makedirs(output_dir, exist_ok=True)

        parser = NukeScriptParser()
        serializer = NukeGraphSerializer(output_dir)

        for i, script_path in enumerate(script_paths):
            with open(script_path, "r", encoding="utf-8") as f:
                script_contents = f.read()

            parsed_script = parser.parse_single_script(script_contents)
            if parsed_script:
                if len(parsed_script) < 10:
                    continue

                base_name = os.path.basename(script_path)
                script_name, _ = os.path.splitext(base_name)
                serializer.serialize_graph(script_name, parsed_script)

            self.status_queue.safe_put(
                TrainingStatus(
                    phase=TrainingPhase.SERIALIZING,
                    progress=float(i / len(script_paths)),
                )
            )

        return output_dir
