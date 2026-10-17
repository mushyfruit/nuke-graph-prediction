import os
import logging
import threading
from typing import Optional, List, Tuple

import torch
from torch_geometric.data import Data

from .main import train_model_gat
from .queue import StatusQueue
from .dataset.vocabulary import Vocabulary
from .dataset.dataset import GraphDataset
from .constants import (
    DirectoryConfig,
    TrainingPhase,
    TrainingStatus,
)
from .utilities import check_state_dict, save_model_checkpoint, check_for_model_on_disk
from ..nuke.parser import NukeScriptParser
from ..nuke.serialization import NukeGraphSerializer

log = logging.getLogger(__name__)


class GNNPipelineManager:
    def __init__(self, model_class, model_config, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._vocab = None

        self._model_class = model_class
        self._model_config = model_config
        self._model_name = model_name

        self.status_queue = StatusQueue(maxsize=100)
        self.plot_queue = StatusQueue(maxsize=100)

        self.training_thread = None
        self._model_lock = threading.RLock()

        self._is_training = threading.Event()

        self.load()

    def get_vocabulary(self):
        return self._vocab

    def load(self) -> bool:
        if not check_for_model_on_disk(self._model_name):
            log.info(f"Model {self._model_name} not found on disk. Skipping load.")
            return False

        if not os.path.exists(DirectoryConfig.VOCAB_PATH):
            raise FileNotFoundError(f"Vocab {DirectoryConfig.VOCAB_PATH} not found")

        model_checkpoint_path = os.path.join(
            DirectoryConfig.MODEL_PATH, f"{self._model_name}_model.pt"
        )
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        state_dict = check_state_dict(checkpoint["state_dict"])

        self._vocab = Vocabulary(DirectoryConfig.VOCAB_PATH)

        self._model = self._model_class.from_checkpoint(
            checkpoint, num_classes=state_dict["prediction_head.8.bias"].numel()
        ).to(self.device)

        self._model.load_state_dict(state_dict)

        self._model.eval()

        return True

    def train(
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

        return TrainingStatus(
            phase=TrainingPhase.SERIALIZING, label="Pipeline started successfully..."
        )

    def _training_thread_target(
        self, file_paths: List[str], memory_allocation: float, enable_fine_tuning: bool
    ) -> None:
        try:
            output_dir = self.parse_and_serialize_scripts(file_paths)
            self.status_queue.safe_put(
                TrainingStatus(
                    phase=TrainingPhase.SERIALIZATION_COMPLETE,
                    label="Script serialization complete!",
                    progress=1.0,
                )
            )

            dataset = GraphDataset(output_dir, force_rebuild=False)

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

            log.info(f"Using: {self._model_config}")

            if enable_fine_tuning and self._model is not None:
                checkpoint = self._model.to_checkpoint()
                training_model = self._model_class.from_checkpoint(
                    checkpoint, num_classes=len(dataset.vocab)
                )
                state = training_model.state_dict()
                for name, value in checkpoint["state_dict"].items():
                    if state[name].shape == value.shape:
                        state[name].copy_(value)
                    else:
                        state[name][: value.shape[0]].copy_(value)
            else:
                training_model = self._model_class(
                    num_features=dataset.num_features,
                    num_classes=len(dataset.vocab),
                    **self._model_config,
                )

            log.info("Training the model...")
            trained_model = train_model_gat(
                dataset,
                training_model,
                memory_fraction=memory_allocation,
                status_queue=self.status_queue,
                plot_queue=self.plot_queue,
            )
            save_model_checkpoint(trained_model, self._model_name)

            # Swap in the trained model.
            # Allow for inference with old model during training.
            with self._model_lock:
                self._model = trained_model
                self._vocab = dataset.vocab

            self.status_queue.safe_put(
                TrainingStatus(
                    phase=TrainingPhase.COMPLETE,
                    label="Finished training the model!",
                    progress=1.0,
                )
            )

        except Exception as e:
            log.exception("Training failed")
            self.status_queue.safe_put(
                TrainingStatus(phase=TrainingPhase.ERROR, error=str(e))
            )
        finally:
            self._is_training.clear()

    def get_training_status(self) -> TrainingStatus:
        if self.status_queue.empty():
            if not self._is_training.is_set():
                return TrainingStatus(
                    phase=TrainingPhase.IDLE, label="Waiting for execution..."
                )
            return TrainingStatus(phase=TrainingPhase.NO_CHANGE)

        return self.status_queue.get_latest()

    def predict(self, pyg_graph_data: Data) -> List[Tuple[str, float]]:
        test_data = pyg_graph_data.to(self.device)

        with self._model_lock, torch.no_grad():
            predictions = self._model(test_data)
            probabilities = torch.nn.functional.softmax(predictions, dim=1)
            top_probs, top_indices = torch.topk(
                probabilities, k=min(10, probabilities.size(1))
            )

            node_predictions = []
            for idx, prob in zip(top_indices[0], top_probs[0]):
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
            try:
                with open(script_path, "r", encoding="utf-8") as f:
                    script_contents = f.read()

                parsed_script = parser.parse_single_script(script_contents)
            except Exception as e:
                log.warning("Skipping %s: %s", script_path, e)
                parsed_script = None
            if parsed_script and len(parsed_script) >= 10:
                base_name = os.path.basename(script_path)
                script_name, _ = os.path.splitext(base_name)
                serializer.serialize_graph(script_name, parsed_script)

            self.status_queue.safe_put(
                TrainingStatus(
                    phase=TrainingPhase.SERIALIZING,
                    progress=float((i + 1) / len(script_paths)),
                )
            )

        return output_dir
