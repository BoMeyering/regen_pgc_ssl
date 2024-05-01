# Model script
import segmentation_models_pytorch as smp
import torch
import numpy as np

def create_smp_model(args):
    if args.encoder_name not in smp.encoders.get_encoder_names():
        raise ValueError(f"Encoder name {args.encoder_name} is not one of the accepted encoders. Please select an encoder from {smp.encoders.get_encoder_names()}")
    if args.model_name == 'unet':
        model = smp.Unet(
            encoder_name=args.encoder_name,
            encoder_depth=args.encoder_depth,
            encoder_weights=args.encoder_weights,
            in_channels=args.in_channels, 
            classes=args.classes
        )
    elif args.model_name == 'unet_plusplus':
        model = smp.UnetPlusPlus(
            encoder_name=args.encoder_name,
            encoder_depth=args.encoder_depth,
            encoder_weights=args.encoder_weights,
            in_channels=args.in_channels, 
            classes=args.classes
        )
    elif args.model_name == 'manet':
        model = smp.MAnet(
            encoder_name=args.encoder_name,
            encoder_depth=args.encoder_depth,
            encoder_weights=args.encoder_weights,
            in_channels=args.in_channels, 
            classes=args.classes
        )
    elif args.model_name == 'linknet':
        model = smp.Linknet(
            encoder_name=args.encoder_name,
            encoder_depth=args.encoder_depth,
            encoder_weights=args.encoder_weights,
            in_channels=args.in_channels, 
            classes=args.classes
        )
    elif args.model_name == 'fpn':
        model = smp.FPN(
            encoder_name=args.encoder_name,
            encoder_depth=args.encoder_depth,
            encoder_weights=args.encoder_weights,
            in_channels=args.in_channels, 
            classes=args.classes
        )
    elif args.model_name == 'pspnet':
        model = smp.PSPNet(
            encoder_name=args.encoder_name,
            encoder_depth=args.encoder_depth,
            encoder_weights=args.encoder_weights,
            in_channels=args.in_channels, 
            classes=args.classes
        )
    elif args.model_name == 'pan':
        model = smp.PAN(
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
            in_channels=args.in_channels, 
            classes=args.classes
        )
    elif args.model_name == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=args.encoder_name,
            encoder_depth=args.encoder_depth,
            encoder_weights=args.encoder_weights,
            in_channels=args.in_channels, 
            classes=args.classes
        )
    elif args.model_name == 'deeplabv3_plus':
        model = smp.DeepLabV3Plus(
            encoder_name=args.encoder_name,
            encoder_depth=args.encoder_depth,
            encoder_weights=args.encoder_weights,
            in_channels=args.in_channels, 
            classes=args.classes
        )
    else:
        raise ValueError(f'args.model_name: {args.model_name} is not a valid model name for the smp framework. Please select an architecture')

    return model
