import os
import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import traceback

from nuke_script.parser import NukeScriptParser
from nuke_script.serialization import NukeGraphSerializer

from model.dataset import NukeGraphBuilder, Vocabulary
from model.model import NukeGATPredictor
from model.main import train_model_gat
from model.constants import MODEL_NAME, MODEL_PATH, VOCAB, MODEL_DATA_FOLDER
from model.utilities import check_state_dict

import torch

from torch_geometric.data import Data


class NukeNodePredictor:
    def __init__(self, model_name: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.vocab = None
        self.model_name = model_name if model_name else MODEL_NAME

        self.load()

    def load(self):
        # Retrieve the checkpoint's path.
        model_path = os.path.join(MODEL_PATH, self.model_name)
        model_checkpoint_path = os.path.join(model_path, f"{self.model_name}_model.pt")
        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"Model {self.model_name} not found")

        vocab_path = os.path.join(model_path, VOCAB)
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab {vocab_path} not found")

        # Load the model checkpoint.
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)

        # Populate the model's stored vocabulary.
        self.vocab = Vocabulary(vocab_path)

        # Instantiate the GAT model.
        self.model = NukeGATPredictor(
            num_features=4,
            num_classes=self.vocab["num_node_types"],
            hidden_channels=checkpoint["hidden_channels"],
            num_layers=checkpoint["num_layers"],
            heads=checkpoint["num_heads"],
            dropout=checkpoint["dropout"],
        ).to(self.device)

        # Restore the state dictionary.
        self.model.load_state_dict(check_state_dict(checkpoint["state_dict"]))

        # Set model into inference mode.
        self.model.eval()

    def train_model(self, directory):
        # parse nuke files -> json files -> dataset
        # dataset = NukeGraphDataset(directory, should_download=False)
        # trained_model = train_model_gat(directory, self.model, memory_fraction=0.5)
        # save_dir = os.path.join(os.path.dirname(__file__), "model", "checkpoints")
        # save_model_checkpoint(trained_model, self.dataset, save_dir, MODEL_NAME)
        pass

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

        for script_path in script_paths:
            with open(script_path, "r", encoding="utf-8") as f:
                script_contents = f.read()

            parsed_script = parser.parse_single_script(script_contents)
            if parsed_script:
                if len(parsed_script) < 10:
                    continue

                base_name = os.path.basename(script_path)
                script_name, _ = os.path.splitext(base_name)
                serializer.serialize_graph(script_name, parsed_script)

        return output_dir


app = FastAPI()

predictor = NukeNodePredictor()
converter = NukeGraphBuilder(predictor.vocab)


@app.post("/train")
async def predict(request: Request):
    """Prediction endpoint"""
    try:
        data = await request.json()
        file_paths = data.get("file_paths", [])
        results = predictor.parse_and_serialize_scripts(file_paths)
        return results
    except Exception as e:
        error_details = {"error": str(e), "traceback": traceback.format_exc()}
        raise HTTPException(status_code=500, detail=f"Training failed: {error_details}")


@app.post("/predict")
async def predict(request: Request):
    """Prediction endpoint"""
    try:
        data = await request.json()

        # get start node name

        start_node_name = data.get("start_node_name")
        nodes = data.get("root", {})["nodes"]
        start_node = nodes[start_node_name]

        # Use the dataset class to convert JSON representations to PYG.
        pyg_graph_data = converter.create_graph_data(data, start_node)
        prediction = predictor.predict(pyg_graph_data)
        result = {"prediction": prediction}

        return result
    except Exception as e:
        error_details = {"error": str(e), "traceback": traceback.format_exc()}
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {error_details}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
