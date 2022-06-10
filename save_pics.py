from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace,
    BooleanOptionalAction,
)
from typing import Tuple

import torch
from torch import Tensor

from torchvision.datasets import VOCSegmentation
import torchvision.transforms as transforms
import torchvision.transforms.functional as func

from torch.utils.data import Dataset, DataLoader

#from datetime import datetime

import numpy as np

import playing


def parse_command_line_args() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        The command-line arguments.

    """
    parser = ArgumentParser(
        description="Image saving example",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model_path", help="Path to model file used", type=str, default="model")

    parser.add_argument("--bs", help="Batch size", type=int, default=2)

    #parser.add_argument(
    #    "--num_layers", help="Number of layers on each side of network. Default in Unet is 5.", 
    #    type=int
    #)

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

    #parser.add_argument(
    #    "--device", help="Device to use (cpu or cuda)", type=str, default="cuda",
    #)

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

from playing import (
    DEVICE,
    get_data_set_and_loader,
    save_pretty_pictures
)

if __name__ == "__main__":
    command_line_args = parse_command_line_args()
    #set_device(command_line_args.device)
    #print(DEVICE)
    validation_set, validation_loader = get_data_set_and_loader(command_line_args, img_set = "val")
    model = torch.load(command_line_args.model_path, map_location=DEVICE)
    save_pretty_pictures(model, validation_set, num_image=5)