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

from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

#from datetime import datetime
from unet import UNet

from PIL import Image
import numpy as np

from playing import get_data_set_and_loader

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

#    parser.add_argument(
#        "--model_root",
#        help="Model name root for saving/loading. Default is 'model'.",
#        type=str,
#        default="model"
#    )

    return parser.parse_args()


def get_single_image(data_set: Dataset):
    """Return a single image + target from dataset.
    Parameters
    ----------
    data_set : Dataset
        Torch dataset
    Returns
    -------
    img : Image tensor
    lab : Target tensor
    """
    n_samples = len(data_set)

    random_index = int(torch.randint(n_samples, (1,)))
    img, lab = data_set[random_index]

    return img, lab

def save_pretty_pictures(model: Module, data_set: Dataset, num_image):
    """Return a single image + target from dataset.
    Parameters
    ----------
    model : Module
        The trained model.
    data_set : Dataset
        Torch dataset
    """
    
    for i in range(num_image):

        with torch.no_grad():
            img, lab = get_single_image(data_set)

            img_gpu = torch.unsqueeze(img,0).to(DEVICE)
            output = model(img_gpu).softmax(dim=1)
        
            output = output.to("cpu")

        img_pred = np.argmax(output, axis=1)
        #img_pred = output.argmax(dim=1).permute(1, 2, 0)
        print(img_pred.shape)
        print(img_pred.float)

        lab = torch.unsqueeze(lab,0)
        img_lab = np.argmax(lab, axis=1)
        img_lab = img_lab.type(torch.float)
        
        orig_img = func.to_pil_image(torch.squeeze(img))
        orig_img.save("orig_img{}.png".format(i))

        pil_pred = seg_to_pil(img_pred)
        pil_pred.save("pred_img{}.png".format(i))

        print(img_lab.shape)
        pil_lab = seg_to_pil(img_lab)
        pil_lab.save("targ_img{}.png".format(i))


def seg_to_pil(img_tensor):
    """Convert model predictions to a pillow image.
    Parameters
    ----------
    img_tensor : Tensor of model predictions for one image. 
        The model to save.

    Returns
    -------
    pil_img : PIL Image
        PIL Image using same palette as targets.
    """
    # Steal the palette from one of the images
    image = Image.open("data/VOCdevkit/VOC2012/SegmentationClass/2011_003271.png")
#     # pixels = (np.array(image.getchannel(0)))
    pal = np.array(image.getpalette()).reshape(256, 3)
    # Apply colour for 255 to value 21, rather than changing the value in the image.
    pal[21, :] = pal[255, :]
    pal = list(pal.flatten())

    pil_img = transforms.ToPILImage()(img_tensor.type(torch.uint8))

#     # print(np.array(pil_img))
    pil_img = pil_img.convert("P")
    pil_img.putpalette(pal)
    
    return pil_img

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to: {DEVICE}.")
    command_line_args = parse_command_line_args()
    #set_device(command_line_args.device)
    #print(DEVICE)
    validation_set, validation_loader = get_data_set_and_loader(command_line_args, img_set = "val")
    model = torch.load(command_line_args.model_path, map_location=DEVICE)
    save_pretty_pictures(model, validation_set, num_image=5)