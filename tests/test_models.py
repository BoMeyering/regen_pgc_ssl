import unittest
import argparse
from unittest.mock import patch, MagicMock, Mock
from src.utils import YamlConfigLoader, ArgsAttributeSetter
from random import randint
import segmentation_models_pytorch as smp

from src.models import create_smp_model

def create_mock_namespace(**kwargs):
    namespace = Mock()
    namespace.__dict__.update(kwargs)
    return namespace

class TestModelCreation(unittest.TestCase):

    def test_invalid_model_name(self):
        invalid_models = ['unten', 'deplabv3_plus', 'fPn', 'namnet', 'tinklet']
        with self.assertRaises(ValueError):
            for name in invalid_models:
                mock_args = create_mock_namespace(model_name=name)

                create_smp_model(mock_args)

    def test_model_instantiation(self):
        model_names = ['unet', 'unet_plusplus', 'manet', 'linknet', 'fpn', 'pspnet', 
                       'pan', 'deeplabv3', 'deeplabv3_plus']
        model_types = [smp.Unet, smp.UnetPlusPlus, smp.MAnet, smp.Linknet, smp.FPN, 
                       smp.PSPNet, smp.PAN, smp.DeepLabV3, smp.DeepLabV3Plus]
        for i, name in enumerate(model_names):
            mock_args = create_mock_namespace(
                model_name=name,
                encoder_name='resnet18',
                encoder_depth=5,
                encoder_weights='imagenet',
                in_channels=3,
                classes=2
            )

            model = create_smp_model(mock_args)

            self.assertIsInstance(model, model_types[i])
    
    @patch('segmentation_models_pytorch.Unet')
    def test_create_model_unet(self, mock_unet):

        mock_args = Mock(
            model_name='unet',
            encoder_name='resnet18',
            encoder_depth=5,
            encoder_weights='imagenet',
            in_channels=3,
            classes=2
        )

        mock_model = Mock()
        mock_unet.return_value = mock_model
        model = create_smp_model(mock_args)

        self.assertEqual(model, mock_model)

    @patch('segmentation_models_pytorch.UnetPlusPlus')
    def test_create_model_unetpp(self, mock_unetpp):
        
        mock_args = Mock(
            model_name='unet_plusplus',
            encoder_name='resnet18',
            encoder_depth=5,
            encoder_weights='imagenet',
            in_channels=3,
            classes=2
        )

        mock_model = Mock()
        mock_unetpp.return_value = mock_model
        model = create_smp_model(mock_args)

        self.assertEqual(model, mock_model)

    @patch('segmentation_models_pytorch.MAnet')
    def test_create_model_manet(self, mock_manet):
        
        mock_args = Mock(
            model_name='manet',
            encoder_name='resnet18',
            encoder_depth=5,
            encoder_weights='imagenet',
            in_channels=3,
            classes=2
        )

        mock_model = Mock()
        mock_manet.return_value = mock_model
        model = create_smp_model(mock_args)

        self.assertEqual(model, mock_model)

    @patch('segmentation_models_pytorch.Linknet')
    def test_create_model_linknet(self, mock_linknet):
        
        mock_args = Mock(
            model_name='linknet',
            encoder_name='resnet18',
            encoder_depth=5,
            encoder_weights='imagenet',
            in_channels=3,
            classes=2
        )

        mock_model = Mock()
        mock_linknet.return_value = mock_model
        model = create_smp_model(mock_args)

        self.assertEqual(model, mock_model)

    @patch('segmentation_models_pytorch.FPN')
    def test_create_model_fpn(self, mock_fpn):
        
        mock_args = Mock(
            model_name='fpn',
            encoder_name='resnet18',
            encoder_depth=5,
            encoder_weights='imagenet',
            in_channels=3,
            classes=2
        )

        mock_model = Mock()
        mock_fpn.return_value = mock_model
        model = create_smp_model(mock_args)

        self.assertEqual(model, mock_model)
    
    @patch('segmentation_models_pytorch.PSPNet')
    def test_create_model_pspnet(self, mock_pspnet):
        
        mock_args = Mock(
            model_name='pspnet',
            encoder_name='resnet18',
            encoder_depth=5,
            encoder_weights='imagenet',
            in_channels=3,
            classes=2
        )

        mock_model = Mock()
        mock_pspnet.return_value = mock_model
        model = create_smp_model(mock_args)

        self.assertEqual(model, mock_model)

    @patch('segmentation_models_pytorch.PAN')
    def test_create_model_pan(self, mock_pan):
        
        mock_args = Mock(
            model_name='pan',
            encoder_name='resnet18',
            encoder_depth=5,
            encoder_weights='imagenet',
            in_channels=3,
            classes=2
        )

        mock_model = Mock()
        mock_pan.return_value = mock_model
        model = create_smp_model(mock_args)

        self.assertEqual(model, mock_model)

    @patch('segmentation_models_pytorch.DeepLabV3')
    def test_create_model_deeplabv3(self, mock_deeplabv3):
        
        mock_args = Mock(
            model_name='deeplabv3',
            encoder_name='resnet18',
            encoder_depth=5,
            encoder_weights='imagenet',
            in_channels=3,
            classes=2
        )

        mock_model = Mock()
        mock_deeplabv3.return_value = mock_model
        model = create_smp_model(mock_args)

        self.assertEqual(model, mock_model)

    @patch('segmentation_models_pytorch.DeepLabV3Plus')
    def test_create_model_deeplabv3_plus(self, mock_deeplabv3_plus):
        
        mock_args = Mock(
            model_name='deeplabv3_plus',
            encoder_name='resnet18',
            encoder_depth=5,
            encoder_weights='imagenet',
            in_channels=3,
            classes=2
        )

        mock_model = Mock()
        mock_deeplabv3_plus.return_value = mock_model
        model = create_smp_model(mock_args)

        self.assertEqual(model, mock_model)

if __name__ == '__main__':
    unittest.main()