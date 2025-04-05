import os
import re
import yaml


def parse_yaml(yaml_file):
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


def parse_cfg(cfg, cfg_path: str):
    """Parses a config file and returns an OmegaConf object."""
    data = parse_yaml(cfg_path)
    for key, value in data.items():
        setattr(cfg, key, value)

    print(cfg)
    return cfg