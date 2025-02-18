import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

import traceback

from model.dataset import NukeGraphBuilder
from model.manager import NukeNodePredictor

from typing import List

app = FastAPI()

predictor = NukeNodePredictor()
converter = NukeGraphBuilder(predictor.vocab)


class TrainingRequest(BaseModel):
    file_paths: List[str]

    @field_validator("file_paths")
    def validate_file_paths(cls, v):
        if not v:
            raise ValueError("At least one file path must be provided")

        for path in v:
            if not os.path.exists(path):
                raise ValueError(f"File not found: {path}")
            if not os.path.isfile(path):
                raise ValueError(f"Not a file: {path}")
            if not os.access(path, os.R_OK):
                raise ValueError(f"File not readable: {path}")
        return v


@app.get("/training_status")
async def get_training_status():
    """Get current training status"""
    training_status = predictor.get_training_status()
    return training_status.to_dict()


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


@app.post("/train")
async def train(request: TrainingRequest):
    """Prediction endpoint"""
    try:
        # Parse and serialize the scripts!
        result = predictor.start_training_pipeline(request.file_paths)
        return result.to_dict()

    except Exception as e:
        error_details = {"error": str(e), "traceback": traceback.format_exc()}
        raise HTTPException(status_code=500, detail=f"Training failed: {error_details}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
