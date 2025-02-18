import os

import threading
import queue

from .nuke_script.parser import NukeScriptParser
from .nuke_script.serialization import NukeGraphSerializer

from .main import train_model_gat
from .dataset import Vocabulary, NukeGraphDataset
from .model import NukeGATPredictor
from .constants import (
    MODEL_NAME,
    MODEL_PATH,
    VOCAB_PATH,
    MODEL_DATA_FOLDER,
    TrainingPhase,
    TrainingStatus,
)
from .utilities import check_state_dict, save_model_checkpoint

import torch
from torch_geometric.data import Data


class NukeNodePredictor:
    def __init__(self, model_name: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.vocab = None
        self.model_name = model_name if model_name else MODEL_NAME

        # Celery workers? Etc.
        self.status_queue = queue.Queue()
        self.training_thread = None
        self.is_running = threading.Event()

        self.load()

    def load(self):
        # Retrieve the checkpoint's path.
        model_checkpoint_path = os.path.join(MODEL_PATH, f"{self.model_name}_model.pt")
        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(
                f"Model {self.model_name} not found at {model_checkpoint_path}"
            )

        if not os.path.exists(VOCAB_PATH):
            raise FileNotFoundError(f"Vocab {VOCAB_PATH} not found")

        # Load the model checkpoint.
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)

        # Populate the model's stored vocabulary.
        self.vocab = Vocabulary(VOCAB_PATH)

        # Instantiate the GAT model.
        self.model = NukeGATPredictor(
            num_features=4,
            num_classes=len(self.vocab),
            hidden_channels=checkpoint["hidden_channels"],
            num_layers=checkpoint["num_layers"],
            heads=checkpoint["num_heads"],
            dropout=checkpoint["dropout"],
        ).to(self.device)

        # Restore the state dictionary.
        self.model.load_state_dict(check_state_dict(checkpoint["state_dict"]))

        # Set model into inference mode.
        self.model.eval()

    def start_training_pipeline(self, file_paths):
        if self.is_running.is_set():
            return TrainingStatus(
                phase=TrainingPhase.SERIALIZING, label="Pipeline already in progress"
            )

        while not self.status_queue.empty():
            self.status_queue.get_nowait()

        self.is_running.set()

        self.training_thread = threading.Thread(
            target=self._training_thread_target, args=(file_paths,)
        )
        self.training_thread.start()

        # Start the process and immediately return.
        return TrainingStatus(
            phase=TrainingPhase.SERIALIZING, label="Pipeline started successfully..."
        )

    def _training_thread_target(self, file_paths):
        try:
            # Start serialization.
            output_dir = self.parse_and_serialize_scripts(file_paths)
            self.status_queue.put(
                TrainingStatus(
                    phase=TrainingPhase.SERIALIZATION_COMPLETE,
                    label="Script serialization complete!",
                    progress=1.0,
                )
            )

            # Begin the training phase.
            dataset = NukeGraphDataset(output_dir)
            self.status_queue.put(
                TrainingStatus(
                    phase=TrainingPhase.TRAINING,
                    label="Starting model training!",
                    progress=0.0,
                )
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            trained_model = train_model_gat(
                dataset,
                self.model,
                memory_fraction=0.5,
                status_queue=self.status_queue,
            )
            save_model_checkpoint(trained_model, MODEL_NAME)

            self.status_queue.put(
                TrainingStatus(
                    phase=TrainingPhase.COMPLETE,
                    label="Finished training the model!",
                    progress=1.0,
                )
            )

        except Exception as e:
            raise
        finally:
            self.is_running.clear()

    def get_training_status(self):
        """Get current training status"""
        if self.status_queue.empty():
            if not self.is_running.is_set():
                return TrainingStatus(
                    phase=TrainingPhase.IDLE, label="Waiting for execution..."
                )
            return TrainingStatus(phase=TrainingPhase.NO_CHANGE)

        return list(self.status_queue.queue)[-1]

    def predict(self, pyg_graph_data: Data):
        test_data = pyg_graph_data.to(self.device)

        # Disable gradient computation.
        with torch.no_grad():
            predictions = self.model(test_data)
            probabilities = torch.nn.functional.softmax(predictions, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=5)

            results = []
            for sample_probs, sample_indices in zip(top_probs, top_indices):
                node_predictions = []
                for idx, prob in zip(sample_indices, sample_probs):
                    node_type = self.vocab.get_type(idx.item())
                    probability = prob.item()
                    node_predictions.append((node_type, probability))
                results.append(node_predictions)

        return results

    def parse_and_serialize_scripts(self, script_paths, output_dir=None):
        if output_dir is None:
            output_dir = MODEL_DATA_FOLDER

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

            self.status_queue.put(
                TrainingStatus(
                    phase=TrainingPhase.SERIALIZING,
                    progress=float(i / len(script_paths)),
                )
            )

        return output_dir
