import os
import glob
import logging
import traceback
from tqdm import tqdm

from parser import NukeScriptParser
from serialization import NukeGraphSerializer

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger()


def get_nuke_script_paths(script_folder_path):
    search_pattern = os.path.join(script_folder_path, "*.nk")
    nuke_scripts = glob.glob(search_pattern)
    return nuke_scripts


def prepare_data_for_training(training_config: dict):
    current_dir = os.path.dirname(os.path.dirname(__file__))
    graph_dir = os.path.join(current_dir, "test_scripts")
    output_dir = os.path.join(current_dir, "test_outputs")

    script_paths = get_nuke_script_paths(graph_dir)
    if not script_paths:
        log.error("Unable to locate any Nuke scripts.")
        return

    if os.path.exists(output_dir):
        if training_config["clear_scripts"]:
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(output_dir)

    parser = NukeScriptParser()
    serializer = NukeGraphSerializer(
        output_dir, include_params=False, remove_passthrough_nodes=True
    )

    # script_paths = [script_paths[21]]

    for script_path in tqdm(script_paths, desc="Parsing scripts"):
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                script_contents = f.read()

            parsed_script = parser.parse_single_script(script_contents)
            if parsed_script:
                if len(parsed_script) < training_config["min_length"]:
                    continue

                base_name = os.path.basename(script_path)
                script_name, _ = os.path.splitext(base_name)
                serializer.serialize_graph(script_name, parsed_script)
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error reading/parsing script {script_path}: {str(e)}")
            continue


def get_config_settings():
    return {"min_length": 20, "clear_scripts": True}


if __name__ == "__main__":
    config = get_config_settings()
    prepare_data_for_training(config)
