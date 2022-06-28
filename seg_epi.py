"""Segmentation of WSI images.

Target images labelled as epithelium (1) or background (0).

"""
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace,
    #BooleanOptionalAction,
)
import json
#from typing import Tuple

import os
from datetime import datetime
from PIL import Image

import torch
from torch import Tensor
from torch.nn import Module, BCELoss
from torch.optim import Adam, Optimizer
#from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader

#from torchvision.datasets import VOCSegmentation
import torchvision.transforms as transforms
import torchvision.transforms.functional as func

from tqdm import tqdm

#from pathlib import Path

from pystain import StainTransformer
from image_dataset import ImageDataset
from unet import UNet
import plot_loss_acc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to: {DEVICE}.")
with torch.no_grad():
    torch.cuda.empty_cache()

#import gc
#gc.collect()

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
        "--epochs", help="Number of epochs to train for.", type=int, default=50,
    )

    parser.add_argument("--bs", help="Batch size", type=int, default=2)

    parser.add_argument("--lr", help="Learning rate.", type=float, default=1e-3)

    parser.add_argument("--wd", help="Weight decay.", type=float, default=0)

    parser.add_argument(
        "--num_classes", help="Number of classes.", type=int, default=2
    )

    parser.add_argument(
        "--num_layers", 
        help="Number of layers on each side of network. Default is 5 to match UNet.",
        type=int, default=5
    )

    parser.add_argument(
        "--loader_workers",
        help="Number of workers for the dataloaders to use.",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--subsample",
        help="Number of training/validation records to use (for testing code). Default is 'all'.",
        type=str,
        default="all"
    )

    parser.add_argument(
        "--train_frac",
        help="Fraction of dataset to use as training data. Default is 0.7",
        type=float,
        default=0.7
    )

    parser.add_argument(
        "--data_dir",
        help="Path to directory containing WSIs. Default is /home/rosie/epithelium_slides/segmentation",
        type=str,
        default="/home/rosie/epithelium_slides/segmentation"
    )

    #parser.add_argument(
    #    "--mask_dir",
    #    help="Path to directory containing segmentation masks. Default is /home/rosie/epithelium_slides/output_masks",
    #    type=str,
    #    default="/home/rosie/epithelium_slides/output_masks"
    #)

    parser.add_argument(
        "--model_root",
        help="Model name root for saving/loading. Default is 'model'.",
        type=str,
        default="model"
    )

    return parser.parse_args()


def change_working_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    my_folder = "UNet_{}".format(timestamp)
    
    os.mkdir(my_folder)
    os.chdir(my_folder)


def write_command_line_args(args: Namespace):
    ''' Write the command line arguments to a file

    Parameters
    ----------
    args : Namespace
        Command-line arguments.
    '''

    with open('command_line_args.txt', 'w') as cmd_file:
        json.dump(args.__dict__, cmd_file, indent=2)


def write_losses_to_file(epoch, training_loss, validation_loss, filename="losses.txt"):
    f = open(filename, "a")
    f.write(f"Epoch {epoch+1} training loss: {training_loss:.3f}, validation loss: {validation_loss:.3f}\n")
    f.close()


def write_acc_to_file(epoch, training_acc, validation_acc, filename="accuracy.txt"):
    f = open(filename, "a")
    f.write(f"Epoch {epoch+1} training accuracy: {training_acc:.3f}, validation accuracy: {validation_acc:.3f}\n")
    f.close()


def get_file_names(file_dir, img_set):
    ''' List of files is without extensions as I'd originally intended to use the same
    list for both images and masks, then realised I had named the mask files too
    stupidly for this to work.

    file_dir needs to contain subdirectories "ImageSets", "images" and "masks"
    '''
    #files = sorted(os.listdir(file_dir))

    img_list = os.path.join(file_dir, "ImageSets", img_set + ".txt")
    mask_list = os.path.join(file_dir, "ImageSets", img_set + "_mask.txt")

    # Images
    with open(img_list) as file_list:
        files = file_list.readlines()

    image_file_names = []

    for file in files:
        image_file_names.append(os.path.join(file_dir, "images", file[:-1] + ".png"))

    # Masks
    with open(mask_list) as file_list:
        files = file_list.readlines()

    mask_file_names = []

    for file in files:
        mask_file_names.append(os.path.join(file_dir, "masks", file[:-1] + ".png"))

    return image_file_names, mask_file_names


def convert_mask_pil_to_tensor(pil_img) -> Tensor:
    """Convert the target mask from B+W pillow image to tensor.

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
    pil_img = pil_img.convert('1')
    grey = func.pil_to_tensor(pil_img).squeeze() / 255
    num_classes = 2
    target = torch.eye(num_classes)[grey.long()].permute(2, 0, 1).float()
    return target

def data_subset(data_set, subsample):
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

    if str.isdigit(subsample):
        if int(subsample) <= len(data_set):
            subsample = torch.randint(len(data_set), (int(subsample),))
            data_set = torch.utils.data.Subset(data_set, subsample)
        else:
            print("Subsample requested is greater than number of records in dataset. Using whole dataset.")
    elif subsample != "all":
        raise ValueError("Subsample requested is not an integer or 'all'. Using whole dataset.")

    return data_set

def split_dataset(dataset, train_frac):
    '''
    Split a dataset into training and validation datasets.
    '''
    train_num = int(train_frac * len(dataset))
    val_num = len(dataset) - train_num

    return torch.utils.data.random_split(dataset, [train_num, val_num])


augment_transforms = transforms.Compose(
    # Set to 572 x 572 to match original UNet paper
    [   
        transforms.RandomRotation((0,180)),
#        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(584, scale=(0.5, 1.0))
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


def get_data_set(data_dir, img_set, subsample):
    '''
    Get training and validation data sets from paths to
    WSI and target mask directories.
    Assumes images and masks will be in the same order when
    sorted alphabetically.
    '''

    if img_set not in ("train", "val", "trainval"):
        raise ValueError (f"Image set option {img_set} is not acceptable.")

    image_file_names, mask_file_names = get_file_names(data_dir, img_set)
    #mask_file_names = get_file_names(mask_dir)

    image_transforms = transforms.Compose(
        [#Image.open,
        StainTransformer(normalise=True, jitter=True, jitter_strength=0.3), 
        #transforms.ToTensor()
        ]
    )
    target_transforms = transforms.Compose(
        [Image.open, 
        convert_mask_pil_to_tensor]
    )

    data_set = ImageDataset(
        inputs = image_file_names,
        image_transforms = image_transforms,
        targets = mask_file_names,
        target_transforms = target_transforms
    )

    data_set = data_subset(data_set, subsample)

#    return split_dataset(data_set, train_frac)
    return data_set

def get_data_loader(data_set, img_set, batch_size, loader_workers):
    """Return a dataloader to use in training/validation.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.
    img_set : Image Set to use. 
        "train" or "val"
    data_set : Dataset
        The dataset to be loaded. 

    Returns
    -------
    data_loader : DataLoader
        The requested dataloader.
    """
    if img_set == 'train' or img_set == 'trainval':
        shuffle_img = True
    elif img_set == 'val':
        shuffle_img = False
    else:
        raise ValueError(f"Image set option {img_set} is not acceptable.")

    data_loader = DataLoader(
        data_set, 
        batch_size=batch_size, 
        shuffle=shuffle_img,
        num_workers=loader_workers,
    )

    return data_loader
    
def calculate_accuracy(predictions, targets):
    ''' Calculate pixel-wise accuracy of predictions. '''

    _, pix_labels = torch.max(predictions, dim=1)
    _, pix_targets = torch.max(targets, dim=1)
    correct = torch.eq(pix_labels,pix_targets).int()
    accuracy = correct.sum() / correct.numel()
    
    return accuracy


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
    running_acc = 0.0
    for images, targets in data_loader:

        ### Include "if" to say if want augmenting. ###
        imgs, targs = data_augmenter(images, targets)

        optimiser.zero_grad()

        imgs, targs = imgs.to(DEVICE), targs.to(DEVICE)

        predictions = model(imgs).softmax(dim=1)

        loss = loss_func(predictions, targs)

        loss.backward()

        optimiser.step()

        running_loss += loss.item()
        running_acc += calculate_accuracy(predictions, targs)
    
    mean_loss = running_loss / len(data_loader)
    accuracy = running_acc / len(data_loader)

    return mean_loss, accuracy


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
    running_acc = 0.0
    with torch.no_grad():
        for imgs, targets in data_loader:
            
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)

            predictions = model(imgs).softmax(dim=1)

            loss = loss_func(predictions, targets)

            running_vloss += loss.item()
            running_acc += calculate_accuracy(predictions, targets)
    
        mean_vloss = running_vloss / len(data_loader) 
        accuracy = running_acc / len(data_loader)

    return mean_vloss, accuracy


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
#    As model now saved in a folder with the timestamp, doesn't need a timestamp.
#    If we do need it, we should re-write to use the same timestamp as the folder.
    model_path = args.model_root
    try:
        torch.save(model, model_path)
    except:
        NotImplementedError("Model not saved correctly.")

    return model_path


def train_model(args: Namespace):
    """Train a segmentation model.

    Parameters
    ----------
    args : Namespace
        The command-line arguments.

    """
    model = UNet(args.num_classes, num_layers=args.num_layers).to(DEVICE)
    optimiser = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_func = BCELoss()

    training_set = get_data_set(args.data_dir, "train", args.subsample)
    validation_set = get_data_set(args.data_dir, "val", args.subsample)

    training_loader = get_data_loader(training_set, "train", args.bs, args.loader_workers)
    validation_loader = get_data_loader(validation_set, "val", args.bs, args.loader_workers)

    print("Training set has {} instances".format(len(training_set)))
    print("Validation set has {} instances".format(len(validation_set)))

    for epoch in tqdm(range(args.epochs)):

        print(f"EPOCH: {epoch+1}")

        mean_loss, accuracy = train_one_epoch(model, training_loader, optimiser, loss_func)
        mean_vloss, vaccuracy = validate_one_epoch(model, validation_loader, loss_func)

        write_losses_to_file(epoch, training_loss = mean_loss, validation_loss = mean_vloss)
        write_acc_to_file(epoch, training_acc = accuracy, validation_acc = vaccuracy)

    model_file = save_model(args, model)
    print(f'Model saved to {model_file}')


if __name__ == "__main__":
    command_line_args = parse_command_line_args()
    change_working_dir()
    write_command_line_args(command_line_args)
    train_model(command_line_args)

    fig = plot_loss_acc.plot_losses_and_accuracies()
    fig.savefig("loss_acc.png", facecolor = "white")
    plt.close(fig)

    