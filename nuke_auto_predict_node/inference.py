import os
import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import traceback


from nuke_script.parser import NukeScriptParser
from nuke_script.serialization import NukeGraphSerializer

from model.dataset import NukeGraphConverter, NukeGraphDataset
from model.model import NukeGATPredictor
from model.main import train_model_gat
from model.constants import MODEL_NAME, MODEL_PATH, VOCAB, MODEL_DATA_FOLDER
from model.utilities import check_state_dict

import torch


class MLModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.vocab = None

        self.node_to_idx_type = None
        self.idx_to_node_type = None

        self.model_name = MODEL_NAME
        self.load_model()

    def load_model(self):
        model_path = os.path.join(MODEL_PATH, self.model_name)
        model_checkpoint_path = os.path.join(model_path, f"{self.model_name}_model.pt")
        vocab_path = os.path.join(model_path, VOCAB)

        if not os.path.exists(model_path):
            return

        # Load checkpoint
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)

        # Load vocabulary
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

        self.node_to_idx_type = self.vocab.get("node_type_to_idx", {})
        self.idx_to_node_type = {v: k for k, v in self.node_to_idx_type.items()}

        self.model = NukeGATPredictor(
            num_features=4,
            num_classes=self.vocab["num_node_types"],
            hidden_channels=checkpoint["hidden_channels"],
            num_layers=checkpoint["num_layers"],
            heads=checkpoint["num_heads"],
            dropout=checkpoint["dropout"],
        ).to(self.device)

        # self.model = torch.compile(self.model)
        self.model.load_state_dict(check_state_dict(checkpoint["state_dict"]))
        self.model.eval()

    def train_model(self, directory):
        # parse nuke files -> json files -> dataset
        # dataset = NukeGraphDataset(directory, should_download=False)
        # trained_model = train_model_gat(directory, self.model, memory_fraction=0.5)
        # save_dir = os.path.join(os.path.dirname(__file__), "model", "checkpoints")
        # save_model_checkpoint(trained_model, self.dataset, save_dir, MODEL_NAME)
        pass

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

    def predict(self, pyg_graph_data):
        test_data = pyg_graph_data.to(self.device)
        with torch.no_grad():
            predictions = self.model(test_data)
            probabilities = torch.nn.functional.softmax(predictions, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=5)

            results = []
            for sample_probs, sample_indices in zip(top_probs, top_indices):
                node_predictions = []
                for idx, prob in zip(sample_indices, sample_probs):
                    node_type = self.idx_to_node_type[idx.item()]
                    probability = prob.item()
                    node_predictions.append((node_type, probability))
                results.append(node_predictions)

        return results


app = FastAPI()

model = MLModel()
converter = NukeGraphConverter(model.node_to_idx_type)


@app.post("/train")
async def predict(request: Request):
    """Prediction endpoint"""
    try:
        data = await request.json()
        file_paths = data.get("file_paths", [])
        results = model.parse_and_serialize_scripts(file_paths)
        return results
    except Exception as e:
        error_details = {"error": str(e), "traceback": traceback.format_exc()}
        raise HTTPException(
            status_code=500, detail=f"Training failed: {error_details}"
        )

@app.post("/predict")
async def predict(request: Request):
    """Prediction endpoint"""
    try:
        data = await request.json()
        pyg_graph_data = converter.convert_json_to_pyg(data)
        prediction = model.predict(pyg_graph_data)
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
