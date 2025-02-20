import os
import queue
import logging
import threading
import traceback
from typing import Optional

import torch
from torch_geometric.data import Data

from .main import train_model_gat
from .dataset import Vocabulary, NukeGraphDataset
from .gat import NukeGATPredictor
from .constants import (
    MODEL_NAME,
    DirectoryConfig,
    TrainingPhase,
    TrainingStatus,
)
from .utilities import check_state_dict, save_model_checkpoint
from ..nuke.parser import NukeScriptParser
from ..nuke.serialization import NukeGraphSerializer

log = logging.getLogger(__name__)


class StatusQueue(queue.Queue):
    def __init__(self, maxsize=100):
        super(StatusQueue, self).__init__(maxsize=maxsize)
        self._lock = threading.Lock()

    def safe_put(self, status: TrainingStatus):
        with self._lock:
            try:
                self.put(status, block=False)
                return True
            except queue.Full:
                try:
                    self.get_nowait()
                    self.put(status, block=False)
                except (queue.Empty, queue.Full):
                    log.warning("Error: failed to put status to queue!")
                    return False

    def safe_get(self):
        try:
            return self.get_nowait()
        except queue.Empty:
            return None

    def get_latest(self) -> Optional[TrainingStatus]:
        try:
            return list(self.queue)[-1]
        except IndexError:
            return None


class NukeNodePredictor:
    def __init__(self, model_name: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self.vocab = None
        self.model_name = model_name if model_name else MODEL_NAME

        self.status_queue = StatusQueue(maxsize=100)

        self.training_thread = None

        self._model_lock = threading.RLock()
        self._is_training = threading.Event()

        self.load()

    def load(self):
        # Retrieve the checkpoint's path.
        model_checkpoint_path = os.path.join(
            DirectoryConfig.MODEL_PATH, f"{self.model_name}_model.pt"
        )
        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(
                f"Model {self.model_name} not found at {model_checkpoint_path}"
            )

        if not os.path.exists(DirectoryConfig.VOCAB_PATH):
            raise FileNotFoundError(f"Vocab {DirectoryConfig.VOCAB_PATH} not found")

        # Load the model checkpoint.
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)

        # Populate the model's stored vocabulary.
        self.vocab = Vocabulary(DirectoryConfig.VOCAB_PATH)

        # Instantiate the GAT model.
        self._model = NukeGATPredictor(
            num_features=4,
            num_classes=len(self.vocab),
            hidden_channels=checkpoint["hidden_channels"],
            num_layers=checkpoint["num_layers"],
            heads=checkpoint["num_heads"],
            dropout=checkpoint["dropout"],
        ).to(self.device)

        # Restore the state dictionary.
        self._model.load_state_dict(check_state_dict(checkpoint["state_dict"]))

        # Set model into inference mode.
        self._model.eval()

    def get_training_model(self, fine_tune=False):
        training_model = NukeGATPredictor(
            num_features=4,
            num_classes=len(self.vocab),
            hidden_channels=self._model.hidden_channels,
            num_layers=self._model.num_layers,
            heads=self._model.heads,
            dropout=self._model.dropout,
        )

        if fine_tune:
            training_model.load_state_dict(self._model.state_dict())

        return training_model

    def start_training_pipeline(
        self, file_paths, memory_allocation, enable_fine_tuning
    ):
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
        self, file_paths, memory_allocation, enable_fine_tuning
    ):
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
            dataset = NukeGraphDataset(output_dir)
            self.status_queue.safe_put(
                TrainingStatus(
                    phase=TrainingPhase.TRAINING,
                    label="Starting model training!",
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

            training_model = self.get_training_model(fine_tune=enable_fine_tuning)
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

    def get_training_status(self):
        """Get current training status"""
        if self.status_queue.empty():
            if not self._is_training.is_set():
                return TrainingStatus(
                    phase=TrainingPhase.IDLE, label="Waiting for execution..."
                )
            return TrainingStatus(phase=TrainingPhase.NO_CHANGE)

        return self.status_queue.get_latest()

    def predict(self, pyg_graph_data: Data):
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
                node_type = self.vocab.get_type(idx.item())
                probability = prob.item()
                node_predictions.append((node_type, probability))

            return node_predictions

    def parse_and_serialize_scripts(self, script_paths, output_dir=None):
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
