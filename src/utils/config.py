# Config/Args Utility Functions
# BoMeyering 2024

import yaml
import os
import pathlib
import argparse
import sys
import logging
from pathlib import Path
from typing import Union
from datetime import datetime
from typing import Tuple, Any

    
class YamlConfigLoader:
    def __init__(self, path: Union[Path, str]) -> None:
        if not isinstance(path, (str, pathlib.Path)):
            raise TypeError(f"'path' argument should be either type 'str' or 'pathlib.Path', not type {type(path)}.")
        if not str(path).endswith(('.yml', '.yaml')):
            raise ValueError(f"path should be a Yaml file ending with either '.yaml' or '.yml'.")
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
        
class ArgsAttributes:
    def __init__(self, args: argparse.Namespace, config: dict) -> None:
        if not isinstance(args, argparse.Namespace):
            raise TypeError(f"'args' should be an argparse.Namespace object.")
        if not isinstance(config, dict):
            raise TypeError(f"'config' should be an dict object.")
        self.args = args
        self.config = config
    
    def append_timestamp_to_run(self):
        now = datetime.now().isoformat(timespec='seconds', sep='_')
        try:
            if hasattr(self.args, 'run_name'):
                self.args.run_name = "_".join((self.args.run_name, now))
            else:
                self.args.run_name = "_".join(('default_run', now))
        except AttributeError as e:
            print(e)
            print(f"Setting default run_name to 'default_run_{now}'")
            setattr(self.args, 'run_name', "_".join(('default_run', now)))

    def set_args_attr(self, check_run_name=True) -> argparse.Namespace:
        """Takes a parsed yaml config file as a dict and adds the arguments to the args namespace.

        Returns:
            argparse.Namespace: The args namespace updated with the configuration parameters.
        """
        for k, v in self.config.items():
            if isinstance(v, dict):
                setattr(self.args, k, argparse.Namespace(**v))
            else:
                setattr(self.args, k, v)
        if check_run_name:
            self.append_timestamp_to_run()

    def validate(self):
        pass

def setup_loggers(args):
    """
    Configures a simple logger to log outputs to the console and the output file.

    Args:
        args (argparse.Namespace): arguments object from the configuration file.
    """
    filename = args.run_name + '.log'
    filepath = Path(args.directories.log_dir) / filename

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.handlers.RotatingFileHandler(filepath, 'a', 1000000, 3)
    stream_handler = logging.StreamHandler(sys.stdout,)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(logging.DEBUG)