from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace,
    #BooleanOptionalAction,
)

import os

import torch
from torch import Tensor
from torch.nn import Module, BCELoss

from image_dataset import ImageDataset
from unet import UNet

from seg_epi import get_data_set, get_data_loader, calculate_dice, validate_one_epoch

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
        "--data_dir",
        help="Path to directory containing WSIs. Default is /mnt/c/Users/rosie/WSL/epithelium_slides",
        type=str,
        default="/mnt/c/Users/rosie/WSL/epithelium_slides"
    )

    parser.add_argument(
        "--model_root",
        help="Model name root for saving/loading. Default is 'model'.",
        type=str,
        default="model"
    )

    return parser.parse_args()

command_line_args = parse_command_line_args()

#os.chdir("../seg_epi/UNet_20220629_151353")

# Get dataset
validation_set = get_data_set(command_line_args.data_dir, "val", "all")

# Load it
validation_loader = get_data_loader(validation_set, "val", 4, 4)

# Get model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to: {DEVICE}.")

model = torch.load("model", map_location=DEVICE)
model.eval()

# Run model for predictions and return dice
loss_func = BCELoss()
mean_vloss, vaccuracy, vdice = validate_one_epoch(model, validation_loader, loss_func)

# Write dice to file
f = open("Dice.txt", "a")
f.write(f"Validation set dice score: {vdice:.3f}\n")
f.close()
