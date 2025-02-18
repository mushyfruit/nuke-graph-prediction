import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

import traceback

from model.dataset import NukeGraphBuilder
from model.manager import NukeNodePredictor

from typing import List, Dict, Any, Optional, Tuple

app = FastAPI()

predictor = NukeNodePredictor()
converter = NukeGraphBuilder(predictor.vocab)


class Node(BaseModel):
    name: str
    node_type: str
    inputs: int
    input_connections: List[str]


class Root(BaseModel):
    name: str
    parent: Optional[str] = None
    nodes: Dict[str, Node]


class PredictionRequest(BaseModel):
    start_node: str
    script_name: str
    root: Root


class PredictionResponse(BaseModel):
    prediction: List[Tuple[str, float]]


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


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Prediction endpoint"""
    try:
        nodes = request.root.nodes
        start_node = nodes[request.start_node].model_dump()
        pyg_graph_data = converter.create_graph_data(request.model_dump(), start_node)
        prediction = predictor.predict(pyg_graph_data)
        return PredictionResponse(prediction=prediction)
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
