import re
from dataclasses import dataclass
from typing import Dict, List

IGNORE_NODES = {"BackdropNode", "StickyNote"}
GROUP_INPUT_NODE = {"LiveInput", "Input"}


@dataclass
class NukeNode:
    name: str
    node_type: str
    inputs: int
    parameters: Dict[str, str]
    script_counters: Dict[str, int]
    input_connections: List[str] = None

    def __post_init__(self):
        if self.input_connections is None:
            self.input_connections = []

        name = self.name
        while self.name in self.script_counters:
            self.script_counters[name] += 1
            self.name = f"{name}_{self.script_counters[name]}"
        self.script_counters[self.name] = 0


class NukeGroup:
    def __init__(self, name, parent=None):
        self.name = name

        self.parent = parent
        self.nodes = {}
        self.stack = []

        self.input_stack = []

    def __len__(self):
        return len(self.nodes)

    def add_node(self, node: NukeNode):
        for _ in range(node.inputs):
            node.input_connections.append(self.stack.pop() if self.stack else None)

        if node.node_type in GROUP_INPUT_NODE:
            self.input_stack.append(node)

        if node.node_type != "Viewer":
            self.nodes[node.name] = node
            self.stack.append(node.name)


class NukeScript:
    def __init__(self):
        self.root_group = NukeGroup("root")
        self.current_group = self.root_group
        self.group_stack = [self.root_group]
        self.groups = {"root": self.root_group}
        self.node_name_counter = {"root": 0}
        self.variables = {}

    def enter_group(self, group_name):
        new_group = NukeGroup(group_name, parent=self.current_group)
        self.group_stack.append(new_group)
        self.current_group = new_group

        self.groups[group_name] = new_group

    def exit_group(self):
        if len(self.group_stack) > 1:
            self.group_stack.pop()
            self.current_group = self.group_stack[-1]

    def __len__(self):
        return sum(len(group) for group in self.groups.values())


class NukeScriptParser:
    GROUP_TYPES = {"LiveGroup", "Group"}

    IGNORE_PARAMS = {"addUserKnob", "onCreate", "knobChanged"}

    def __init__(self):
        self.node_start = re.compile(r"\s*([\w.-]+)(?:\s+\$(\w+))?\s*{\s*(.*)$")
        self.node_param = re.compile(r"\s*(\w+)\s+(.*)")
        self.set_var = re.compile(r"\s*set\s+(\w+)\s+\[stack\s+0\]")
        self.push_var = re.compile(r"\s*push\s+\$(\w+)")
        self.push_null_value = re.compile(r"\s*push\s+0")

    def parse_single_script(self, content):
        self.multiline_param = None
        self.current_param_value = []
        self.brace_depth = 0
        script = NukeScript()
        current_node = None
        current_params = {}
        ignored_depth = 0

        for line in content.splitlines():
            line = line.rstrip()
            if ignored_depth:
                ignored_depth += self.get_brace_depth(line)
                continue
            if self.multiline_param is not None:
                self.handle_node_parameter(line, current_params)
                continue
            if not line or line.lstrip().startswith("#"):
                continue

            if not current_node:
                if line.strip() == "end_group":
                    script.exit_group()
                    continue

                if line.lstrip().startswith(("Root {", "define_window_layout_xml ", "add_layer ")):
                    ignored_depth = self.get_brace_depth(line)
                    continue
                node_match = self.node_start.fullmatch(line)
                if node_match:
                    current_node = node_match.group(1)
                    if current_node in IGNORE_NODES:
                        ignored_depth = self.get_brace_depth(line)
                        current_node = None
                    elif current_node == "clone":
                        source_name = script.variables[node_match.group(2)]
                        source = next(
                            group.nodes[source_name]
                            for group in script.groups.values()
                            if source_name in group.nodes
                        )
                        current_node = source.node_type
                        current_params = source.parameters.copy()
                        current_params.pop("inputs", None)
                    if current_node:
                        params = node_match.group(3)
                        if self.get_brace_depth(line) == 0:
                            self.handle_node_parameter(
                                params.rstrip().removesuffix("}").rstrip(), current_params
                            )
                            line = "}"
                        else:
                            self.handle_node_parameter(params, current_params)
                            continue
                    else:
                        continue

            if current_node:
                if line.strip() != "}":
                    self.handle_node_parameter(line, current_params)
                    continue
                current_params.setdefault("name", current_node)
                node = NukeNode(
                    name=current_params["name"],
                    node_type=current_node,
                    inputs=sum(map(int, current_params.get("inputs", "1").split("inputs")[0].strip().split("+"))),
                    script_counters=script.node_name_counter,
                    parameters=current_params,
                )
                script.current_group.add_node(node)
                if node.node_type in self.GROUP_TYPES:
                    script.enter_group(node.name)
                current_node = None
                current_params = {}
                continue

            set_match = self.set_var.match(line)
            if set_match:
                script.variables[set_match.group(1)] = (
                    script.current_group.stack[-1] if script.current_group.stack else None
                )
                continue
            push_match = self.push_var.match(line)
            if push_match:
                script.current_group.stack.append(
                    script.variables.get(push_match.group(1))
                )
                continue
            if self.push_null_value.match(line):
                script.current_group.stack.append(None)

        return script

    def handle_node_parameter(self, line, current_params):
        if self.multiline_param is not None:
            self.brace_depth += self.get_brace_depth(line)
            self.current_param_value.append(line.strip())
            if self.brace_depth == 0:
                if self.multiline_param not in self.IGNORE_PARAMS:
                    current_params[self.multiline_param] = "\n".join(
                        self.current_param_value
                    )
                self.current_param_value = []
                self.multiline_param = None
            return

        param_match = self.node_param.match(line)
        if not param_match:
            return

        param_name = param_match.group(1)
        param_value = param_match.group(2)

        if self.get_brace_depth(line) > 0:
            self.multiline_param = param_name
            self.brace_depth = self.get_brace_depth(line)
            self.current_param_value = [param_value]

        if (
            not self._is_empty_param(param_name, param_value)
            and param_name not in self.IGNORE_PARAMS
        ):
            current_params[param_name] = param_value

    def get_brace_depth(self, line):
        line = re.sub(r"\\.", "", line)
        open_braces = line.count("{")
        close_braces = line.count("}")
        return open_braces - close_braces

    def _is_empty_param(self, param_name, value):
        if param_name == "name":
            return False

        if not value:
            return True
        if value in ('""', "''", "{}", "[]", "null", "none", "None"):
            return True

        return False
