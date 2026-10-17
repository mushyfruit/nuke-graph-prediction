# NodeScout

A Nuke plugin that predicts the next node with a GATv2 graph neural network trained on your `.nk` scripts.

Requires **Nuke 17.0v3, Linux x86-64, Python 3.11**, internet access for initial setup, and a writable plugin directory. The server runs separately and reuses Nuke's bundled libtorch/CUDA to keep the venv small.

1. Add this repository's root to `NUKE_PATH` before launching Nuke:

   ```sh
   export NUKE_PATH="/path/to/nuke-graph-prediction${NUKE_PATH:+:$NUKE_PATH}"
   ```

2. Launch Nuke normally. `menu.py` starts setup in the background on first use. Follow `node_scout/checkpoints/logs/server.log` until setup finishes and the server starts.
3. Open **Pane > NodeScout**. In **Training**, choose your scripts and train a model. In **Prediction**, select a node and press **Ctrl + Shift + T**, or enable **Predict on Selection**. Double-click a recommendation to create it.

Plugin logs: `node_scout/checkpoints/logs/auto-predict.log`. The server stops when Nuke exits. Set a distinct `AUTO_PREDICT_PORT` before launching each simultaneous Nuke instance (default `8080`).

To set up manually, run `node_scout/bin/create-venv.sh /path/to/Nuke17.0v3 [venv-directory]`. `TORCH_WHEEL=/path/to/torch-2.7.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl` reuses a downloaded wheel. Rebuild the venv when changing Nuke installations.
