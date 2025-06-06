import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import traceback

from core.model.dataset import NukeGraphBuilder
from core.model.manager import NukeNodePredictor

from typing import List, Dict, Optional, Tuple

app = FastAPI()

predictor = NukeNodePredictor()
converter = NukeGraphBuilder(load_from_disk=True)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    root: Root


class PredictionResponse(BaseModel):
    prediction: List[Tuple[str, float]]
    start_node: str


class TrainingRequest(BaseModel):
    file_paths: List[str]
    memory_allocation: float
    enable_fine_tuning: bool

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


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/training_status")
async def get_training_status():
    """Get current training status"""
    training_status = predictor.get_training_status()
    return training_status.to_dict()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Prediction endpoint"""
    try:
        log.info(f"Predicting node {request.start_node}")
        nodes = request.root.nodes
        start_node = nodes[request.start_node].model_dump()
        pyg_graph_data = converter.create_graph_data(
            request.model_dump(),
            start_node,
            filter_graphs=False,
            ensure_valid_vocabulary=True,
        )
        prediction = predictor.predict(pyg_graph_data)
        return PredictionResponse(prediction=prediction, start_node=start_node)
    except Exception as e:
        error_details = {"error": str(e), "traceback": traceback.format_exc()}
        log.error(error_details)
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {error_details}"
        )


@app.post("/train")
async def train(request: TrainingRequest):
    """Prediction endpoint"""
    try:
        # Parse and serialize the scripts!
        log.info("Received a training request!")
        result = predictor.start_training_pipeline(
            request.file_paths, request.memory_allocation, request.enable_fine_tuning
        )
        return result.to_dict()

    except Exception as e:
        error_details = {"error": str(e), "traceback": traceback.format_exc()}
        log.error(error_details)
        raise HTTPException(status_code=500, detail=f"Training failed: {error_details}")


if __name__ == "__main__":
    import uvicorn

    log.info(f"Starting the server on {sys.argv[1]}:{sys.argv[2]}")
    try:
        uvicorn.run(
            app, host=str(sys.argv[1]), port=int(sys.argv[2]), log_level="warning"
        )
    except Exception as e:
        log.error(f"Server failed to start: {e}")
