import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, AugmentedCarvanaDataset
from utils.dice_score import dice_loss
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms():
    return A.Compose([
        A.Resize(height=64, width=64), 
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
# dir_checkpoint = Path('./checkpoints/')

# dir_img = Path('././data_poligono/imgs/')
# dir_mask = Path('././data_poligono/masks/')
# dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/data_deeplabv3_poligono')


# dir_img = Path('././data_geral/imgs/')
# dir_mask = Path('././data_geral/masks/')
# dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/data_geral_deeplabv3_poligono')

# dir_img = Path('././data_combined/imgs/')
# dir_mask = Path('././data_combined/masks/')
# dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/data_combined')

# dir_img = Path('././data_estrada_sondagem/imgs/')
# dir_mask = Path('././data_estrada_sondagem/masks/')
# dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/data_estrada_sondagem_deeplabv3_camila')

# dir_img = Path('././data_estrada_sondagem_splits/split_01/imgs/')
# dir_mask = Path('././data_estrada_sondagem_splits/split_01/masks/')
# dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/data_estrada_sondagem_splits_split_1')

# dir_img = Path('././data_estrada_sondagem_splits/split_02/imgs/')
# dir_mask = Path('././data_estrada_sondagem_splits/split_02/masks/')
# dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/data_estrada_sondagem_splits_split_2')

# dir_img = Path('././data_estrada_sondagem_splits/split_03/imgs/')
# dir_mask = Path('././data_estrada_sondagem_splits/split_03/masks/')
# dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/data_estrada_sondagem_splits_split_3')

# dir_img = Path('././data_estrada_sondagem_splits/split_04/imgs/')
# dir_mask = Path('././data_estrada_sondagem_splits/split_04/masks/')
# dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/data_estrada_sondagem_splits_split_4')

# dir_img = Path('././data_estrada_sondagem_splits/split_05/imgs/')
# dir_mask = Path('././data_estrada_sondagem_splits/split_05/masks/')
# dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/data_estrada_sondagem_splits_split_5')

# dir_img = Path('././data_estrada_sondagem_splits/split_06/imgs/')
# dir_mask = Path('././data_estrada_sondagem_splits/split_06/masks/')
# dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/data_estrada_sondagem_splits_split_6')

# dir_img = Path('././data_estrada_sondagem_splits/split_07/imgs/')
# dir_mask = Path('././data_estrada_sondagem_splits/split_07/masks/')
# dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/data_estrada_sondagem_splits_split_7')

# dir_img = Path('././data_estrada_sondagem_splits/split_08/imgs/')
# dir_mask = Path('././data_estrada_sondagem_splits/split_08/masks/')
# dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/data_estrada_sondagem_splits_split_8')

# dir_img = Path('././data_estrada_sondagem_splits/split_09/imgs/')
# dir_mask = Path('././data_estrada_sondagem_splits/split_09/masks/')
# dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/data_estrada_sondagem_splits_split_9')

# dir_img = Path('/media/igor/LTS/Flavio/data_areas_antropicas/imgs')
# dir_mask = Path('/media/igor/LTS/Flavio/data_areas_antropicas/masks')
# dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/areas_antropicas')

# dir_img = Path('/media/igor/LTS/Flavio/data_lajedo/imgs')
# dir_mask = Path('/media/igor/LTS/Flavio/data_lajedo/masks')
# dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/lajedo')

# dir_img = Path('/media/igor/LTS/Flavio/data_mata_baixa/imgs')
# dir_mask = Path('/media/igor/LTS/Flavio/data_mata_baixa/masks')
# dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/mata_baixa')

dir_img = Path('/media/igor/LTS/Flavio/data_vegetacao_rupestre_aberta/imgs')
dir_mask = Path('/media/igor/LTS/Flavio/data_vegetacao_rupestre_aberta/masks')
dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/vegetacao_rupestre_aberta')

# dir_img = Path('././data_estrada_sondagem/imgs/')
# dir_mask = Path('././data_estrada_sondagem/masks/')
# dir_checkpoint = Path('/media/igor/LTS/Flavio/checkpoints/data_deeplabv3')


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        augment_times: int = 4,
):
    # 1. Create dataset
    # try:
        # dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    if augment_times == 0:
        dataset = BasicDataset(dir_img, dir_mask, img_scale)
    else:
        transforms = get_transforms()
        augmented_datasets = [AugmentedCarvanaDataset(images_dir=dir_img, mask_dir=dir_mask, scale=1, transforms=transforms) for _ in range(augment_times)]
        dataset = ConcatDataset(augmented_datasets)
    # except (AssertionError, RuntimeError, IndexError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.25, patience=3)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                # assert images.shape[1] == model.n_channels, \
                #     f'Network has been defined with {model.n_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    
                    if "out" in masks_pred:
                        masks_pred = masks_pred["out"]

                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                        # Evaluate each class
                        class_eval = []
                        for i in range(model.n_classes):
                            mask_i = F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2)[:, i].float()
                            pred_i = F.softmax(masks_pred, dim=1)[:, i].float()
                            class_eval.append(dice_loss(pred_i, mask_i, multiclass=False))
                            
                            
                

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()


                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch,
                    
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        for i, class_loss in enumerate(class_eval):
                            logging.info(f'class {i} loss: {class_loss.item()}')
                            experiment.log({
                                f'class {i} loss': class_loss.item(),
                                'step': global_step,
                                'epoch': epoch
                            })
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            if isinstance(dataset, ConcatDataset):
                mask_values = []
                for ds in dataset.datasets:
                    if hasattr(ds, 'mask_values'):
                        mask_values.extend(ds.mask_values)
                state_dict['mask_values'] = list(set(mask_values))  # Remover duplicatas
            else:
                state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--model', '-m', type=str, default='unet', choices=['unet', 'deeplabv3'],
                        help='Choose the model to use: "unet" or "deeplabv3"')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--augment-times', '-a', type=int, default=4, help='Number of times to augment the dataset')

    return parser.parse_args()

def get_deeplabv3_model(n_classes):
    # model = deeplabv3_resnet50(weights='DeepLabV3_ResNet50_Weights.DEFAULT')
    model = deeplabv3_resnet101(weights='DeepLabV3_ResNet101_Weights.DEFAULT')
    model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))
    model.n_classes = n_classes
    return model

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # Escolher o modelo com base no argumento
    if args.model == 'unet':
        model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model == 'deeplabv3':
        model = get_deeplabv3_model(n_classes=args.classes)

    model = model.to(memory_format=torch.channels_last)


    # logging.info(f'Network:\n'
                #  f'\t{model.n_channels} input channels\n'
                #  f'\t{model.n_classes} output channels (classes)\n'
                #  f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            augment_times=args.augment_times
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            augment_times=args.augment_times
        )
