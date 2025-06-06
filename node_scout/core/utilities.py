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


def get_all_parameters(node):
    blacklist_knobs = {"xpos", "ypos"}

    parameter_dict = {}
    for knob in node.knobs():
        if knob in blacklist_knobs:
            continue

        k = node[knob]
        knob_value = k.value()

        try:
            if k.defaultValue() == knob_value:
                continue
        # Not all knobs define a 'defaultValue()'
        except Exception as e:
            pass

        if not isinstance(knob_value, (str, float, int)):
            continue

        if knob_value == "":
            continue

        parameter_dict[knob] = node[knob].value()

    return parameter_dict
