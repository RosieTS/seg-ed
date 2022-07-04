from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace,
    BooleanOptionalAction,
)

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset

import torchvision.transforms as transforms
import torchvision.transforms.functional as func

#from unet import UNet
from seg_epi import get_data_set


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

    parser.add_argument("--image_path", help="Path to image file directory", type=str, default="")

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


def epi_seg_to_pil(img_tensor):
    """Convert epithelium masks / predictions to a pillow image.
    Parameters
    ----------
    img_tensor : Tensor 
        Pixel values for one segmentation image (mask or prediction) - one-hot encoded.
        Needs to have form (1, 2, H, W).

    Returns
    -------
    pil_img : PIL Image
        PIL Image
    """
    
    img_tensor = np.argmax(img_tensor, axis=1)
    img_tensor = img_tensor * 255

    pil_img = transforms.ToPILImage()(img_tensor.type(torch.uint8))
        
    return pil_img


def mask_outlines(mask_image):
    '''
    Convert a segmentation mask (filled) to a mask of outlines

    Parameters
    ----------
        mask_image : PIL image, greyscale (0 to 255)
    Returns
    -------
        edges : np array (0 or 255)
    '''

    #image = np.array(Image.open(filename))
    image = np.array(mask_image)
    kernel = np.ones((3, 3), 'uint8')
    #image = cv.dilate(image, kernel)
    #image = cv.erode(image, kernel)

    edges = cv.Canny(image, 1, 254)
    edges = cv.dilate(edges, kernel)

    return edges

def draw_mask_on_orig(orig_image, edges):
    '''
    Draw segmentation mask/prediction outlines on the original image
    (in yellow-green)
    
    Parameters
    ----------
        orig_image : PIL image, colour
        edges : np array (H x W, values 0 or 255)
    Returns
    -------
        orig_with_mask : np array (H x W x 3)
    '''
    
    orig_image = np.array(orig_image)
    idx = edges[:, :] == 255
    orig_with_mask = orig_image.copy()
    orig_with_mask[idx] = (127, 255, 0)

    return orig_with_mask


def save_pretty_pictures(model: Module, data_set: Dataset, num_image):
    """Return a single image + target from dataset.
    Parameters
    ----------
    model : Module
        The trained model.
    data_set : Dataset
        Torch dataset
    """

    fig, axes = plt.subplots(nrows=2, ncols=num_image, sharex=True, sharey=True,
                         figsize=(4*num_image, 8))

    for i in range(num_image):

        img, lab = get_single_image(data_set)

        with torch.no_grad():
            img_gpu = torch.unsqueeze(img,0).to(DEVICE)

            output = model(img_gpu).softmax(dim=1)
        
            img_pred = output.to("cpu")

        img_lab = torch.unsqueeze(lab,0)

        pil_pred = epi_seg_to_pil(img_pred)
        pil_lab = epi_seg_to_pil(img_lab)

        edges = mask_outlines(pil_lab)
        edges_pred = mask_outlines(pil_pred)

        orig_img = func.to_pil_image(torch.squeeze(img))
        
        orig_lab = draw_mask_on_orig(orig_img, edges)
        orig_pred = draw_mask_on_orig(orig_img, edges_pred)

        axes[0][i].imshow(orig_lab)
        axes[0][i].axis('off')    

        #axes[0].imshow(targ_img, alpha = 0.2)
        axes[1][i].imshow(orig_pred)
        axes[1][i].axis('off')    

        #axes[1].imshow(pred_img, alpha = 0.2)

    plt.savefig("results1.png")
    #orig_img.save("orig_img{}.png".format(i))
    #pil_pred.save("pred_img{}.png".format(i))
    #pil_lab.save("targ_img{}.png".format(i))

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to: {DEVICE}.")
    command_line_args = parse_command_line_args()

    data_set = get_data_set(command_line_args.image_path, "val", subsample = "all")

    model = torch.load(command_line_args.model_path, map_location=DEVICE)
    model.eval()
    save_pretty_pictures(model, data_set, num_image=5)