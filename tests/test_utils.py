import unittest
import argparse
from unittest.mock import MagicMock
from src.utils import YamlConfigLoader, ArgsAttributeSetter

class TestYamlConfigLoader(unittest.TestCase):
    def test_init_with_invalid_path_type(self):
        with self.assertRaises(TypeError):
            YamlConfigLoader(123)

    def test_init_with_nonexistent_path(self):
        with self.assertRaises(FileNotFoundError):
            YamlConfigLoader("nonexistent_file.yml")

    def test_init_with_invalid_file_extension(self):
        with self.assertRaises(ValueError):
            YamlConfigLoader("config.txt")

    def test_load_config(self):
        # Mocking open and yaml.safe_load
        with unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data="key: value")):
            # config_loader = YamlConfigLoader(MagicMock())
            config_loader = YamlConfigLoader("tests/test_config.yaml")
            config = config_loader.load_config()
            self.assertEqual(config, {"key": "value"})

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
