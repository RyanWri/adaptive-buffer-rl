from pathlib import Path
import yaml


def load_config():
    """
    Load a YAML configuration file into a dictionary.

    :return: Configuration as a dict.
    :raises FineNotFoundError: If the file does not exist.
    :raises ValueError: If the YAML is invalid.
    """

    project_root = Path(__file__).resolve().parents[2]
    config_file = project_root / "config.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"Config file '{config_file}' not found!")

    with config_file.open('r', encoding='utf-8') as cfg_file:
        try:
            cfg = yaml.safe_load(cfg_file)
            return cfg if cfg is not None else {}
        except yaml.YAMLError as ex:
            raise ValueError(f"Error while parsing YAML file: {ex}")
