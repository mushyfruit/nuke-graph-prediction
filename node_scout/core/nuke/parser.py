import re
import logging
import traceback
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List

IGNORE_NODES = {"BackdropNode", "StickyNote"}
GROUP_INPUT_NODE = {"LiveInput", "Input"}

log = logging.getLogger(__name__)


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

            # If we've seen this name before, increment counter and modify name
            if self.name in self.script_counters:
                self.script_counters[self.name] += 1
                self.name = f"{self.name}_{self.script_counters[self.name]}"
            else:
                self.script_counters[self.name] = 0


class NukeGroup:
    def __init__(self, name, num_inputs, parent=None):
        self.name = name
        self.num_inputs = num_inputs

        self.parent = parent
        self.nodes = {}
        self.stack = []

        # Keep track of inputs for flattening later.
        self.input_stack = []
        self.variables = {}
        self._node_counter = 0

    def __len__(self):
        return self._node_counter

    def add_node(self, node: NukeNode):
        if node.inputs > 0:
            for _ in range(node.inputs):
                input_node = self.stack.pop()
                node.input_connections.insert(0, input_node)

        # Keep track of group inputs.
        if node.node_type in GROUP_INPUT_NODE:
            self.input_stack.append(node)

        # Don't track viewer as they are terminal nodes.
        if node.node_type != "Viewer":
            self._node_counter += 1
            self.nodes[node.name] = node
            self.stack.append(node.name)


class NukeScript:
    def __init__(self):
        self.root_group = NukeGroup("root", 0)
        self.current_group = self.root_group
        self.group_stack = [self.root_group]
        self.groups = {"Root": self.root_group}
        self.node_name_counter = {}

    def enter_group(self, group_name, group_inputs):
        """Enter a new group context"""
        new_group = NukeGroup(group_name, group_inputs, parent=self.current_group)
        self.group_stack.append(new_group)
        self.current_group = new_group

        self.groups[group_name] = new_group

    def exit_group(self, verbose=False):
        """Exit current group context"""

        if verbose:
            log.info([x for x in self.group_stack[-1].nodes.keys()])
            log.info(f"Exiting {self.group_stack[-1].name}")

        if len(self.group_stack) > 1:
            self.group_stack.pop()
            self.current_group = self.group_stack[-1]

    def __len__(self):
        return sum([len(group) for group in self.groups.values()])


class NukeScriptParser:
    GROUP_TYPES = {"LiveGroup", "Group"}

    # Skip callback knobs
    IGNORE_PARAMS = {"addUserKnob", "onCreate", "knobChanged"}

    def __init__(self, skip_dots=True):
        self.skip_dots = skip_dots

        # Allow leading white space.
        # Group depth dictates # of preceding white spaces.
        self.node_start = re.compile(r"^(\s*)(\w+)\s*{\s*$")

        self.node_param = re.compile(r"\s*(\w+)\s+(.*)")
        self.set_var = re.compile(r"\s*set\s+(\w+)\s+\[stack\s+0\]")
        self.push_var = re.compile(r"\s*push\s+\$(\w+)")
        self.push_null_value = re.compile(r"\s*push\s+0")

        self.collecting_multiline = False
        self.multiline_param = None
        self.current_param_value = []
        self.brace_depth = 0

    def parse_script(self, contents, script_filter=None):
        if not contents:
            log.info("No contents found!")
            return

        parsed_scripts = {}
        for project_name, project_data in contents.items():
            for script_name, script_contents in tqdm(
                project_data.items(), desc="Parsing {0} scripts".format(project_name)
            ):
                if script_filter is not None:
                    if script_filter not in script_name:
                        continue

                try:
                    parsed_scripts[script_name] = self.parse_single_script(
                        script_contents
                    )
                except Exception:
                    log.info(
                        f"Error parsing script {script_name}: {traceback.format_exc()}"
                    )
                    return

        return parsed_scripts

    def parse_single_script(self, content):
        """Parse a single Nuke script content"""
        script = NukeScript()

        # Track current node being built
        current_node = None
        current_params = {}

        ignoring = False

        for i, line in enumerate(content.split("\n")):
            # Only strip trailing whitespace, preserve leading
            line = line.rstrip()

            # Skip an empty or irrelevant lines!
            if not line or line.startswith("#") or line.startswith("Root"):
                continue

            # Handle group stack, exiting if 'end_group' syntax is encountered.
            if line.strip() == "end_group":
                script.exit_group()
                continue

            # Check for new node definitions!
            node_match = self.node_start.match(line)
            if node_match:
                indentation = node_match.group(1)
                matched_node_name = node_match.group(2)
                if current_node is None:
                    ignoring = False

                    if matched_node_name in IGNORE_NODES:
                        ignoring = True
                        continue

                    current_node = matched_node_name
                    current_params = {}

                    continue

            # If we're ignoring a node, skip through to next node.
            if ignoring:
                if line.strip() == "}":
                    ignoring = False
                continue

            # Check for node definition completion.
            if line.strip() == "}" and current_node and not self.multiline_param:
                # Finish adding the current node to the graph.
                try:
                    # If node doesn't define a "name" param in .nk,
                    # then it falls back to the node_type for name?
                    try:
                        num_inputs = int(current_params.get("inputs", 1))
                    except ValueError:
                        # Possible to define inputs with expression (e.g. 1+1)
                        expr_str = current_params["inputs"]
                        a, b = map(int, expr_str.split("+"))
                        num_inputs = a + b

                    node = NukeNode(
                        name=current_params["name"],
                        node_type=current_node,
                        inputs=num_inputs,
                        script_counters=script.node_name_counter,
                        parameters=current_params,
                    )

                    script.current_group.add_node(node)

                except Exception:
                    log.error("---------------------")
                    log.error(content.split("\n")[i - 15 : i + 5])
                    log.error(f"Line:  {i}")
                    log.error(f"Current group: {script.current_group.name}")
                    log.error(f"Current group stack: {script.current_group.stack}")
                    log.error(f"Current node: {current_node}")
                    log.error(current_params.keys())
                    log.error(f"Current param?: {self.multiline_param}")
                    raise

                # Enter new group after creation!
                if node.node_type in self.GROUP_TYPES:
                    script.enter_group(node.name, node.inputs)

                # Clear the current node/parameter references.
                current_node = None
                current_params = {}
                continue

            # Check for the node 'set' syntax.
            set_match = self.set_var.match(line)
            if set_match:
                var_name = set_match.group(1)
                if script.current_group.stack:
                    script.current_group.variables[var_name] = (
                        script.current_group.stack[-1]
                    )

            # Check for push variable to push nodes onto stack.
            push_match = self.push_var.match(line)
            if push_match:
                var_name = push_match.group(1)
                if var_name in script.current_group.variables:
                    script.current_group.stack.append(
                        script.current_group.variables[var_name]
                    )
                continue

            # For empty inputs, Nuke scripts will use "push 0"
            null_push_match = self.push_null_value.match(line)
            if null_push_match:
                script.current_group.stack.append(None)
                continue

            # Otherwise, attempting to check for node parameters!
            if current_node:
                self.handle_node_parameter(line, current_params)

        return script

    def handle_node_parameter(self, line, current_params):
        if self.multiline_param is not None:
            self.brace_depth += self.get_brace_depth(line)
            if self.brace_depth == 0:
                current_params[self.multiline_param] = "\n".join(
                    self.current_param_value
                )
                self.current_param_value = []
                self.multiline_param = None
            else:
                self.current_param_value.append(line.strip())
            return

        param_match = self.node_param.match(line)
        if not param_match:
            return

        param_name = param_match.group(1)
        param_value = param_match.group(2)

        # Check if this parameter starts a new block
        if self.multiline_param is None and self.get_brace_depth(line) > 0:
            self.multiline_param = param_name
            self.brace_depth = self.get_brace_depth(line)

        if (
            not self._is_empty_param(param_name, param_value)
            and param_name not in self.IGNORE_PARAMS
        ):
            current_params[param_name] = param_value

    def get_brace_depth(self, line):
        open_braces = line.count("{")
        close_braces = line.count("}")
        return open_braces - close_braces

    def _is_empty_param(self, param_name, value):
        """Check if a parameter value should be considered empty"""
        # Nodes can be named whatever.
        if param_name == "name":
            return False

        if not value:
            return True
        if value in ('""', "''", "{}", "[]", "null", "none", "None"):
            return True

        # Handle whitespace-only strings
        if isinstance(value, str) and not value.strip():
            return True

        return False
