import os
import shutil
import yaml
import json
import importlib
import asyncio
from typing import List, Dict


class BasePipeline:

    def __init__(
        self,
        working_dir: str = ".working_dir",
        **kwargs,
    ):
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def init_from_config(
        cls,
        config_path: str,
        working_dir: str = ".working_dir",
    ):
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        components = {}
        for key, value in config.items():
            class_path = value["class_path"]
            init_args = value["init_args"]
            module_name, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            components[key] = getattr(module, class_name)(**init_args)

        return cls(**components, working_dir=working_dir)
