#!/bin/bash
set -euo pipefail
trap 'echo "NodeScout setup failed; the inference server was not started." >&2' ERR

NUKE_DIR="$(cd "${1:?Usage: create-venv.sh NUKE_DIR [VENV_DIR]}" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${2:-$SCRIPT_DIR/../inference_venv}"
VENV_PYTHON="$VENV_DIR/bin/python3"
unset PYTHONPATH PYTHONHOME PYTORCH_INSTALL
export PYTHONNOUSERSITE=1

LIBRARIES=(libtorch libtorch_cpu libtorch_cuda libc10 libc10_cuda
           libtorch_global_deps libcaffe2_nvrtc)
for name in "${LIBRARIES[@]}"; do
    if [ ! -f "$NUKE_DIR/$name.so" ]; then
        echo "NodeScout requires Nuke's bundled $name.so in $NUKE_DIR." >&2
        exit 1
    fi
done
if ! grep -qF "/.conan/data/PyTorch/2.7.1/" "$NUKE_DIR/libc10.so"; then
    echo "NodeScout requires Nuke's bundled libtorch 2.7.1; refusing an incompatible graft." >&2
    exit 1
fi

"$NUKE_DIR/python3.11" -m venv --clear "$VENV_DIR"
"$VENV_PYTHON" -m pip install --no-cache-dir --no-deps \
    "${TORCH_WHEEL:-torch==2.7.1+cu128}" --index-url https://download.pytorch.org/whl/cu128
"$VENV_PYTHON" -m pip install --no-cache-dir --no-deps \
    filelock==4.2.1 typing_extensions==4.16.0 sympy==1.13.3 mpmath==1.3.0 \
    networkx==3.6.1 Jinja2==3.1.6 MarkupSafe==3.0.4 fsspec==2026.9.0
"$VENV_PYTHON" -m pip install --no-cache-dir -r "$SCRIPT_DIR/../../inference_requirements.txt"

PATCH_ELF="$(command -v patchelf || true)"
if [ -z "$PATCH_ELF" ]; then
    "$VENV_PYTHON" -m pip install --no-cache-dir --no-deps patchelf==0.17.2.4
    PATCH_ELF="$VENV_DIR/bin/patchelf"
fi
TORCH_LIB="$VENV_DIR/lib/python3.11/site-packages/torch/lib"
for name in "${LIBRARIES[@]}"; do
    ln -sf "$NUKE_DIR/$name.so" "$TORCH_LIB/$name.so"
done
rm "$TORCH_LIB/libtorch_cuda_linalg.so"
# Foundry disables cuSPARSELt; NodeScout uses ordinary GAT/scatter operations.
"$PATCH_ELF" --remove-needed libcusparseLt.so.0 "$TORCH_LIB/libtorch_python.so"

export LD_LIBRARY_PATH="$NUKE_DIR" TORCH_USE_RTLD_GLOBAL=1 TORCH_COMPILE_DISABLE=1
"$VENV_PYTHON" - <<'PY'
import torch
from torch_geometric.nn import GATv2Conv
assert torch.__version__ == "2.7.1+cu128", "NodeScout requires torch 2.7.1+cu128 bindings"
for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
    model = GATv2Conv(2, 2).to(device)
    model(torch.ones(2, 2, device=device), torch.tensor([[0], [1]], device=device)).sum().backward()
print(f"NodeScout setup verified: torch {torch.__version__}, CUDA {torch.cuda.is_available()}")
PY
touch "$VENV_DIR/.nodescout-ready"
if [ "$#" -gt 2 ]; then
    shift 2
    exec "$@"
fi
