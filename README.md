# nuke graph next-node prediction
A Nuke plugin that leverages a locally trained machine-learning model to predict the next node based on your unique workflows.

Includes a full framework for training Graph Attention Networks (GATs) specifically for Nuke, featuring a script parser and serializers, GAT implementations, and a lightweight FastAPI server for performing model inference in a separate process.

### Features
- **ML Prediction**s: Utilizes a Graph Attention Network (GATv2) for intelligent node recommendations.
- **Graph Serialization**: Converts `.nk` scripts into structured and flattened graph data for model training and inference.
- **End-to-End Training & Inference**: Supports model training and predictions via a custom panel within Nuke.

### Getting Started
1. **Copy the contents of `init.py` and `menu.py`:**
    ```python
    # menu.py
    from nuke_auto_predict_node import nuke_auto_predict_node
    nuke_auto_predict_node.on_startup()
    ```
2. **Generate virtual environment for inference:**
    - When launching the plugin for the first time, `bin/create-venv.sh` will be executed.
    - The dependencies for the inference server will be automatically started.
3. **Open the Prediction Panel:**
    - Navigate to **Windows > Custom > NodeRecommendation** in Nuke.
4. **Train a local model:**
    - Under the "Training" tab, click **Browse** and select a folder containing your Nuke scripts.
    - Click Train to begin training the model on your scripts.
5. **Get Predictions:**
    - Once the model is finished training, click on the **Prediction** tab.
    - Press **Ctrl + Shift + T** when selecting a node to perform a next-node prediction.
      - Or enable **Predict on Selection** to perform predictions automatically upon selecting a node.
    - The predicted Node Type and confidence score will be displayed.
    - Double-click a node-type entry to automatically create it in the node graph.

### Logs & Debugging

Logs are stored in:
```
checkpoints/logs/auto-predict.log
```
Inference logs are stored in:
```
checkpoints/logs/server.log
```

### Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.