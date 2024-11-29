import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
import rasterio

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        if isinstance(output, dict):
            output = output['out']
        
        output = output.cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--arquitecture', '-a', default='unet',
                        help='Arquitecture of the model')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(input_folder, output_folder):
    # Given a list of input filenames, return a list of output filenames in the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    return [os.path.join(output_folder, os.path.basename(f)) for f in input_folder]


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def get_deeplabv3_model(n_classes):
    model = deeplabv3_resnet101(weights='DeepLabV3_ResNet101_Weights.DEFAULT')
    model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))
    model.n_classes = n_classes
    return model

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = os.listdir(args.input)
    out_files = get_output_filenames(in_files, os.path.abspath(args.output))

    if args.arquitecture == 'unet':
        net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.arquitecture == 'deeplabv3':
        net = get_deeplabv3_model(args.classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        if not filename.endswith('.tiff'):
            continue

        filename = os.path.join(args.input, filename)
        logging.info(f'Predicting image {filename} ...')
        # img = Image.open(filename)
        # Open the image with rasterio
        with rasterio.open(filename) as src:
            img = src.read([1, 2, 3])
            img = np.moveaxis(img, 0, -1)
            img = Image.fromarray(img)

            mask = predict_img(net=net,
                            full_img=img,
                            scale_factor=args.scale,
                            out_threshold=args.mask_threshold,
                            device=device)
            
            # Save the profile of the image to the mask
            profile = src.profile.copy()
            profile.update({
                'count': 1,
                'dtype': 'uint8'
            })

            # Save the mask
            if not args.no_save:
                with rasterio.open(out_files[i], 'w', **profile) as dest:
                    dest.write(mask, 1)
            

                # out_filename = out_files[i]
                # result = mask_to_image(mask, mask_values)
                # result.save(out_filename)
                logging.info(f'Mask saved to {out_files[i]}')

            if args.viz:
                logging.info(f'Visualizing results for image {filename}, close to continue...')
                plot_img_and_mask(img, mask)