import os
import sys


# TODO: Improve detection.
def detect_current_dcc() -> str:
    exe = os.path.basename(sys.executable).lower()
    if "nuke" in exe:
        return "nuke"
    elif "hfs" in sys.executable:
        return "houdini"
    else:
        raise RuntimeError("Unsupported platform.")
