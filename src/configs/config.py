"""This is configuration file to load data, training model, hyperparameters"""
import os
from typing import Dict

import yaml
from sklearn import set_config

CONFIG_PATH = "src/configs/config.yaml"

set_config(transform_output="pandas")


def load_config() -> Dict:
    """Loads config.yaml file in configs folder"""
    with open(os.path.join(CONFIG_PATH)) as file:
        config = yaml.safe_load(file)
    return config
