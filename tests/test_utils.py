import unittest
import argparse
from unittest.mock import patch, MagicMock
from src.utils import YamlConfigLoader, ArgsAttributeSetter
from random import randint

class TestYamlConfigLoader(unittest.TestCase):
    
    def test_init_with_invalid_path_type(self):
        with self.assertRaises(TypeError):
            for i in range(10):
                YamlConfigLoader(randint(100, 4000))

    def test_init_with_nonexistent_path(self):
        with self.assertRaises(FileNotFoundError):
            YamlConfigLoader("nonexistent_file.yml")

    def test_init_with_invalid_file_extension(self):
        with self.assertRaises(ValueError):
            YamlConfigLoader("config.txt")

    @patch("builtins.open", new_callable=MagicMock)
    @patch("yaml.safe_load", return_value={"parameter": 0.0001})
    def test_load_config(self, mock_open, mock_safe_load):
        # Mocking open and yaml.safe_load
        config_loader = YamlConfigLoader("tests/test_config.yaml")
        config = config_loader.load_config()
        self.assertEqual(config, {"parameter": 0.0001})

class TestArgsAttributeSetter(unittest.TestCase):
    def test_init_with_invalid_args_type(self):
        with self.assertRaises(TypeError):
            ArgsAttributeSetter("not_a_namespace", {})

    def test_init_with_invalid_config_type(self):
        with self.assertRaises(TypeError):
            ArgsAttributeSetter(argparse.Namespace(), "not_a_dict")

    def test_set_args_attr(self):
        args = argparse.Namespace()
        args.some_arg = None
        config = {"some_arg": "some_value"}
        args_setter = ArgsAttributeSetter(args, config)
        updated_args = args_setter.set_args_attr()
        self.assertEqual(updated_args.some_arg, "some_value")

if __name__ == "__main__":
    unittest.main()
