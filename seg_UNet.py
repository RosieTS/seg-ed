"""Segmentation training example.

Apart from U-net, all the below is taken from the tutorial below:
https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

I used the below to figure out that the target images are in "p" mode,
which means "palettised". Which is basically one channel with numbers
representing palette items, so I can just use that like a greyscale.

from PIL import Image
image = Image.open("data/VOCdevkit/VOC2012/SegmentationClass/2011_003271.png")
image.mode

The pixel classes are 1-20 for type of object, plus 0 for background
and 255 for void/unlabelled.
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html
"""
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace,
    BooleanOptionalAction,
)
import json
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module, BCELoss
from torch.optim import Adam, Optimizer


from torchvision.datasets import VOCSegmentation
import torchvision.transforms as transforms
import torchvision.transforms.functional as func

from torch.utils.data import Dataset, DataLoader

from datetime import datetime
from tqdm import tqdm
import os
#from pathlib import Path

from unet import UNet


#DEVICE = "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to: {DEVICE}.")


def parse_command_line_args() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        The command-line arguments.

    """
    parser = ArgumentParser(
        description="Segmentation training example",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--epochs", help="Number of epochs to train for.", type=int, default=10,
    )

    parser.add_argument("--bs", help="Batch size", type=int, default=2)

    # NB This is not used in the call to the optimiser - it is not doing anything.
    parser.add_argument("--lr", help="Learning rate.", type=float, default=1e-3)

    parser.add_argument(
        "--num_classes", help="Number of classes.", type=int, default=22
    )

    parser.add_argument(
        "--num_layers", help="Number of layers on each side of network. Default is 5 to match UNet.", 
        type=int, default=5
    )

    parser.add_argument(
        "--data_download",
        type=bool,
        default=False,
        action=BooleanOptionalAction,
        help="Should we download/unpack the data? Default is False.",
    )

    parser.add_argument(
        "--loader_workers",
        help="Number of workers for the dataloaders to use.",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--subsample", 
        help="Number of training/validation records to use (for testing code). Default is 'all'.", 
        type=str,
        default="all"
    )

    parser.add_argument(
        "--model_root",
        help="Model name root for saving/loading. Default is 'model'.",
        type=str,
        default="model"
    )

    return parser.parse_args()


def write_command_line_args(args: Namespace):
    ''' Write the command line arguments to a file

    Parameters
    ----------
    args : Namespace
        Command-line arguments.
    '''

    with open('command_line_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def write_losses_to_file(epoch, training_loss, validation_loss, filename="losses.txt"):
    f = open(filename, "a")
    f.write(f"Epoch {epoch+1} training loss: {training loss:.3f}, validation loss: {validation loss:.3f}")
    f.close()

def convert_target_pil_to_tensor(pil_img) -> Tensor:
    """Convert the target mask from pillow image to tensor.

    Parameters
    ----------
    pil_img : PIL.Image
        The segmentation mask as a pillow image.

    Returns
    -------
    target : Tensor
        The segmentation mask as a (C, H, W) Tensor.

    Notes
    -----
    If:
        grey[i, j] = 0, target[:, i, j] = [1, 0, 0, ...]
        grey[i, j] = 1, target[:, i, j] = [0, 1, 0, ...]
        grey[i, j] = 2, target[:, i, j] = [0, 0, 1, ...],

        etc.

    """
    grey = func.pil_to_tensor(pil_img).squeeze()
    grey[grey == 255] = 21
    num_classes = 22
    target = torch.eye(num_classes)[grey.long()].permute(2, 0, 1).float()
    return target


img_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        # Resizing because images all different sizing. Could instead pad or make custom collate.
        # Set to 572 x 572 to match original UNet paper
        transforms.Resize([572, 572]),
    ]
)

target_transforms = transforms.Compose(
    # Set to 572 x 572 to match original UNet paper
    [convert_target_pil_to_tensor, transforms.Resize([572, 572])]
)


def get_data_set_and_loader(args: Namespace, img_set) -> Tuple[Dataset, DataLoader]:
    """Return a dataset and dataloader to use in training/validation.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.
    img_set : Image Set to use. 
        "train" or "val"

    Returns
    -------
    data_set : Dataset
        The requested dataset.
    data_loader : DataLoader
        The requested dataloader.

    """
    if img_set == 'train':
        shuffle_img = True
    elif img_set == 'val':
        shuffle_img = False
    else:
        raise ValueError(f"Image set option {img_set} is not acceptable.")


    data_set = VOCSegmentation(
        "../data",
        #image_set="train",
        image_set=img_set,
        download=args.data_download,
        transform=img_transforms,
        target_transform=target_transforms,
    )

    data_set = data_subset(args, data_set)

    data_loader = DataLoader(
        data_set, 
        batch_size=args.bs, 
        shuffle=shuffle_img,
        num_workers=args.loader_workers,
    )

    return data_set, data_loader

def data_subset(args: Namespace, data_set):
    """Return a subsample of records from a dataset.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.
    data_set : Dataset.
        The data set to sample from.
    
    Returns
    -------
    data_set : Dataset
        The subsampled dataset.
    """

    if str.isdigit(args.subsample):
        if int(args.subsample) <= len(data_set):
            subsample = torch.randint(len(data_set), (int(args.subsample),))
            data_set = torch.utils.data.Subset(data_set, subsample)
        else:
            print("Subsample requested is greater than number of records in dataset. Using whole dataset.")
    elif args.subsample != "all":
        raise ValueError("Subsample requested is not an integer or 'all'. Using whole dataset.")

    return data_set

def train_model(args: Namespace):
    """Train a segmentation model.

    Parameters
    ----------
    args : Namespace
        The command-line arguments.

    """
    model = UNet(args.num_classes, num_layers=args.num_layers).to(DEVICE)
    optimiser = Adam(model.parameters(), weight_decay=1e-3)
    loss_func = BCELoss()

    training_set, training_loader = get_data_set_and_loader(args, img_set = "train")
    validation_set, validation_loader = get_data_set_and_loader(args, img_set = "val")

    print("Training set has {} instances".format(len(training_set)))
    print("Validation set has {} instances".format(len(validation_set)))

    for epoch in tqdm(range(args.epochs)):

        print(f"EPOCH: {epoch+1}")

        running_loss = train_one_epoch(model, training_loader, optimiser, loss_func)
        running_vloss = validate_one_epoch(model, validation_loader, loss_func)

        write_losses_to_file(epoch, training_loss = running_loss, validation_loss = running_vloss)

    model_file = save_model(args, model)
    print(f'Model saved to {model_file}')


augment_transforms = transforms.Compose(
    # Set to 572 x 572 to match original UNet paper
    [   
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(572, scale=(0.25, 1.0))
    ]
)

def data_augmenter(images, targets):
    """Transform a batch of images and targets using a random resized crop
    
    Parameters
    ----------
    images : set of images
        Original images.
    targets : set of targets
        Original targets.

    Returns
    ----------
    images_new : set of images
        Images after transformation.
    targets_New : set of targets
        Targets after transformation.
    """

    img_concat = torch.cat((images, targets), dim=1)
    augmented = augment_transforms(img_concat)
    images_new = augmented[:, :3, :, :]
    targets_new = augmented[:, 3:, :, :]

    return images_new, targets_new


def train_one_epoch(
    model: Module, data_loader: DataLoader, optimiser: Optimizer, loss_func: Module,
) -> float:
    """Train `model` for a single epoch.

    Vanilla Pytorch training loop.

    Parameters
    ----------
    model : Module
        The model to train.
    data_loader : DataLoader
        The data to train on.

    Returns
    -------
    running_loss : float
        The total loss for this epoch.

    """
    running_loss = 0.0
    for imgs, targets in data_loader:

        ### Include "if" to say if want augmenting. ###
        imgs_aug, targets_aug = data_augmenter(imgs, targets)
        imgs = torch.cat((imgs, imgs_aug), dim=0)
        targets = torch.cat((targets, targets_aug), dim=0)

        optimiser.zero_grad()

        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)

        predictions = model(imgs).softmax(dim=1)

        loss = loss_func(predictions, targets)

        loss.backward()

        optimiser.step()

        running_loss += loss.item()

    return running_loss

def validate_one_epoch(
    model: Module, data_loader: DataLoader, loss_func: Module,
) -> float:
    """Validate `model` for a single epoch.

    Parameters
    ----------
    model : Module
        The model to train.
    data_loader : DataLoader
        The validation data loader.

    Returns
    -------
    running_vloss : float
        The total loss for this epoch.

    """
# We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    with torch.no_grad():
        for imgs, targets in data_loader:
            
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)

            predictions = model(imgs).softmax(dim=1)

            loss = loss_func(predictions, targets)

            running_vloss += loss.item()

    return running_vloss


def save_model(args: Namespace, model):
    """Save the model.
    Parameters
    ----------
    args : Namespace
        Command-line arguments.
    model : Module
        The model to save.

    Returns
    -------
    model_path : Path name
        Path where model is saved.
    """
#    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#    model_path = "{}_{}".format(args.model_root, timestamp)
    try:
#        torch.save(model, model_path)
#       As model now saved in a folder with the timestamp, doesn't need a timestamp.
#       If we do need it, we should re-write to use the same timestamp as the folder.
        torch.save(model, model_root)
    except:
        NotImplementedError("Model not saved correctly.")

    return model_path

def change_working_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    my_folder = "UNet_{}".format(timestamp)
    
    os.mkdir(my_folder)
    os.chdir(my_folder) 


if __name__ == "__main__":
    command_line_args = parse_command_line_args()
    change_working_dir()
    write_command_line_args(command_line_args)
    #print("DEVICE is now: {}".format(DEVICE))
    train_model(command_line_args)
    