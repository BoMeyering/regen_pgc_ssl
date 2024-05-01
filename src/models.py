# Model creation script
# BoMeyering 2024
import segmentation_models_pytorch as smp
import argparse
import torch

def create_smp_model(args: argparse.Namespace) -> torch.nn.Module:
    """Creates an smp Pytorch model

    Args:
        args (argparse.Namespace): The argparse namespace from a config file

    Raises:
        ValueError: If args.model.encoder_name is not listed in smp.encoders.get_encoder_names().
        ValueError: If args.model.model_name does not match any of the specified architectures.

    Returns:
        torch.nn.Module: A model as a pytorch module
    """
    if args.model.encoder_name not in smp.encoders.get_encoder_names():
        raise ValueError(f"Encoder name {args.model.encoder_name} is not one of the accepted encoders. Please select an encoder from {smp.encoders.get_encoder_names()}")
    if args.model.model_name == 'unet':
        model = smp.Unet(
            encoder_name=args.model.encoder_name,
            encoder_depth=args.model.encoder_depth,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    elif args.model.model_name == 'unet_plusplus':
        model = smp.UnetPlusPlus(
            encoder_name=args.model.encoder_name,
            encoder_depth=args.model.encoder_depth,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    elif args.model.model_name == 'manet':
        model = smp.MAnet(
            encoder_name=args.model.encoder_name,
            encoder_depth=args.model.encoder_depth,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    elif args.model.model_name == 'linknet':
        model = smp.Linknet(
            encoder_name=args.model.encoder_name,
            encoder_depth=args.model.encoder_depth,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    elif args.model.model_name == 'fpn':
        model = smp.FPN(
            encoder_name=args.model.encoder_name,
            encoder_depth=args.model.encoder_depth,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    elif args.model.model_name == 'pspnet':
        model = smp.PSPNet(
            encoder_name=args.model.encoder_name,
            encoder_depth=args.model.encoder_depth,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    elif args.model.model_name == 'pan':
        model = smp.PAN(
            encoder_name=args.model.encoder_name,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    elif args.model.model_name == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=args.model.encoder_name,
            encoder_depth=args.model.encoder_depth,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    elif args.model.model_name == 'deeplabv3_plus':
        model = smp.DeepLabV3Plus(
            encoder_name=args.model.encoder_name,
            encoder_depth=args.model.encoder_depth,
            encoder_weights=args.model.encoder_weights,
            in_channels=args.model.in_channels, 
            classes=args.model.num_classes
        )
    else:
        raise ValueError(f'args.model.model_name: {args.model.model_name} is not a valid model name for the smp framework. Please select an architecture')

    return model
