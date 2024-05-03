# Config/Args Utility Functions
# BoMeyering 2024

import yaml
import os
import pathlib
import argparse
from pathlib import Path
from typing import Union
from datetime import datetime
from typing import Tuple, Any

    
class YamlConfigLoader:
    def __init__(self, path: Union[Path, str]) -> None:
        if not isinstance(path, (str, pathlib.Path)):
            raise TypeError(f"'path' argument should be either type 'str' or 'pathlib.Path', not type {type(path)}.")
        if not str(path).endswith(('.yml', '.yaml')):
            raise ValueError(f"path should be a Yaml file ending with either '.yamll' or '.yml'.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File path at {path} does not exist. Please specify a different path")
        
        self.path = path

    def load_config(self) -> dict:
        """Reads a yaml config file at path and returns a dictionary of config arguments.

        Returns:
            dict: A dictionary of key/value pairs for the arguments in the config file.
        """
        with open(self.path, 'r') as file:
            return yaml.safe_load(file)
        
class ArgsAttributeSetter:
    def __init__(self, args: argparse.Namespace, config: dict) -> None:
        if not isinstance(args, argparse.Namespace):
            raise TypeError(f"'args' should be an argparse.Namespace object.")
        if not isinstance(config, dict):
            raise TypeError(f"'config' should be an dict object.")
        self.args = args
        self.config = config

    def set_nested_key(self, args: argparse.Namespace, keys: Tuple[str, str], value: Any='default_value') -> None:

        if not hasattr(args, keys[0]):
            setattr(args, keys[0], argparse.Namespace())
        namespace = getattr(args, keys[0])
        if not hasattr(namespace, keys[1]):
            setattr(namespace, keys[1], value)
        
        self.args = args

        return self.args


    def set_args_attr(self) -> argparse.Namespace:
        """Takes a parsed yaml config file as a dict and adds the arguments to the args namespace.

        Returns:
            argparse.Namespace: The args namespace updated with the configuration parameters.
        """
        for k, v in self.config.items():
            if isinstance(v, dict):
                setattr(self.args, k, argparse.Namespace(**v))
            else:
                setattr(self.args, k, v)
        
        if self.args.general.run_name:
            now = datetime.now().isoformat(timespec='seconds', sep='_')
            self.args.general.run_name = "_".join((self.args.general.run_name, now))
        else:
            self.args.general.run_name
        return self.args

def increment_training_run(args):
    run = args.general.training_run
    checkpoint_dir = args.directories.chkpt_dir

    current_runs = os.listdir(checkpoint_dir)