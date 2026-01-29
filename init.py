"""Nuke init.py hook.

This file is sourced by Nuke on startup.

If you need Nuke's embedded Python to find a specific PyTorch install, set the
`PYTORCH_INSTALL` environment variable to the target `site-packages` path.

We intentionally do *not* hardcode a machine-specific path here.
"""

import os

# Example:
# os.environ.setdefault("PYTORCH_INSTALL", "/path/to/site-packages")
