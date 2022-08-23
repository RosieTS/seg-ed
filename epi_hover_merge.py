"""This code takes epithelium prediction masks at 10x resolution and
merges them with nucleus prediction masks at 40x resolution to return
(and save as a .pkl file) a data frame containing a list of all epithelial
nuclei and their properties. 

Requires as input:
    Path to directory containing 10x resolution images. There must be an equivalent folder
        at the same level with the masks, i.e. ../masks
    Path to directory containing 40x resolution images
    Path to directory containing HoVerNet output directories (json, mat, overlay)
    Path to (torch) model used for predictions

Outputs: 
    Pickle file : epi_nuc_data.pkl
    Single sample image to confirm everything looks OK: PREFIX_epi_nuc.png

"""


import sys

#sys.path.append('../')
import numpy as np
import pandas as pd
import math
import os
import glob
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
import json
import openslide

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace,
    BooleanOptionalAction,
)

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pystain import StainTransformer

import unet
from image_dataset import ImageDataset


def parse_command_line_args() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        The command-line arguments.

    """
    parser = ArgumentParser(
        description="Program to generate and merge epithelial masks at 10x magnification \
            with nucleus segmentation data at 40x magnification",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--image_path",
        help="Path to directory containing biopsy image tile used to train epithelilum segmentation \
            model. Can include file root / wildcard \
            to return subset of images, but in this case must be enclosed in ''. \
            Default is '/home/ret58/rds/hpc-work/epithelium_slides/images/'",
        type=str,
        default="/home/ret58/rds/hpc-work/epithelium_slides/images/"
    )

    parser.add_argument(
        "--hi_res_image_path",
        help="Path to directory containing biopsy image tile. Can include file root / wildcard \
            to return subset of images, but in this case must be enclosed in ''. \
            Default is '/home/ret58/rds/hpc-work/epithelium_slides_hi_res/images/'",
        type=str,
        default="/home/ret58/rds/hpc-work/epithelium_slides_hi_res/images/"
    )

    parser.add_argument(
        "--model_path",
        help="Model path + name for epithelium mask model to be used. Default is 'model'",
        type=str,
        default="model"
    )

    parser.add_argument(
        "--hov_path",
        help="Path to directory containing hovernet output directories with nucleus predictions. \
            List of individual files to use will be derived from image file names. \
            Default is '/home/ret58/rds/hpc-work/hover_net/consep_hi_res/'",
        type=str,
        default="/home/ret58/rds/hpc-work/hover_net/consep_hi_res/"
    )

    parser.add_argument(
        "--bs",
        help="Batch size for running predictions.",
        type=int,
        default=8
    )

    parser.add_argument(
        "--lw",
        help="Loader workers for running predictions.",
        type=int,
        default=4
    )

    parser.add_argument(
        "--predictions",
        type=bool,
        default=True,
        action=BooleanOptionalAction,
        help="Should we run the model for the predictions? Default is True. If False will run from \
            .npy files"
    )

    return parser.parse_args()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to: {DEVICE}.")

def get_image_file_names(image_file_path):
    """Use image file path to return list of image file names.
    
    Parameters
    ----------
    image_file_path : str
        Full path to image file directory, or alternatively a directory
        + wildcard search ending in .png
    
    Returns
    -------
    image_file_names : str list
        List of paths to image files
    """
    
    if image_file_path[-4:] == ".png":
        image_file_names = glob.glob(image_file_path)
    elif "." not in image_file_path:
        glob_pattern = os.path.join(image_file_path, '*', '*')
        image_file_names = glob.glob(glob_pattern)      
    else:
        raise ValueError("image_path should be a directory or a set of .png files.")

    image_file_names.sort()
    
    return image_file_names


def get_basename(image_file_name):
    """Get the basename of an image from its file name.
    
    Parameters
    ----------
    image_file_name : str
        Path to image file
    
    Returns
    -------
    Image base name (path removed, no suffix)
    """

    return os.path.basename(image_file_name).split('.')[0]


def get_json_name(basename, hov_path):
    """Return a json filename using the image basename and HoVerNet file path
    
    Parameters
    ----------
    basename : str
        Image base name
    hov_path : Path to HoVerNet directory
    
    Returns
    -------
    Path to hovernet json file
    """

    return os.path.join(hov_path, 'json', basename + '.json')


def get_mat_file_name(basename, hov_path):
    """Return a mat filename using the image basename and HoVerNet file path
    
    Parameters
    ----------
    basename : str
        Image base name
    hov_path : Path to HoVerNet directory
    
    Returns
    -------
    Path to hovernet mat file
    """

    return os.path.join(hov_path, 'mat', basename + '.mat')


def get_dataset(image_file_names):
    '''
    Get data set from list of file names.

    Parameters
    ----------
    image_file_names : List
        List of file names containing images
    
    Returns
    -------
    data_set : Dataset
        Dataset of all the images, normalized
    '''

    image_transforms = transforms.Compose(
        [StainTransformer(normalise=True)]
        )
    
    data_set = ImageDataset(
        inputs = image_file_names,
        image_transforms = image_transforms
    )

    return data_set


def get_data_loader(data_set, batch_size, loader_workers):
    """Return a dataloader to use in training/validation.

    Parameters
    ----------
    data_set : Dataset
        The dataset to be loaded.
    batch_size : int
        Batch size to be used.
    loader_workers : int
        Number of loader workers to be used.

    Returns
    -------
    data_loader : DataLoader
        The requested dataloader.
    """
    data_loader = DataLoader(
        data_set, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=loader_workers,
    )

    return data_loader


def generate_predictions(model: Module, imgs):
    """Generate epithelium predictions from trained model +
    input images at 10x resolution. 
    
    Parameters
    ----------
    model : Module
        Trained model to use
    imgs : Tensor
        Mini-batch of images

    Returns
    -------
    imgs_pred : Tensor
        Mini-batch of images
    """

    imgs_gpu = imgs.to(DEVICE)

    outputs = model(imgs_gpu).softmax(dim=1)
    imgs_pred = outputs.to("cpu")

    imgs_pred = 255 * torch.argmax(imgs_pred, dim=1)

    return imgs_pred


def write_preds_to_file(predictions, batch_num, batch_size,
        image_file_names):
    """Save predictions to a temporary file in case of crashes. 
    
    Parameters
    ----------
    predictions : Tensor
        Mini-batch of predictions
    batch_num : int
        Running count of which batch we're on
    batch_size : int
        Mini-batch size
    image_file_names : List
        List of paths to image files
    """
    
    if os.path.isdir("tmp") == False:
        os.mkdir("tmp")

    start_file = batch_num * batch_size

    for i in range(predictions.size(dim=0)):
       
        basename = get_basename(image_file_names[start_file + i])
        #basename = os.path.basename(image_file_names[start_file + i]).split('.')[0]
        pred_file_name = "tmp/" + basename + ".npy"

        epi_mask = predictions[i].numpy()
        np.save(pred_file_name, epi_mask)


def output_predictions(model: Module, data_loader, image_file_names):
    """Cycle through dataset and generate epithelium predictions from trained model +
    input images at 10x resolution; write these to temporary files
    
    Parameters
    ----------
    model : Module
        Trained model to use
    data_loader : Dataloader
        Data loader to use
    image_file_names : List
        List of paths to image files

    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to: {DEVICE}.")

    with torch.no_grad():

        for batch_num, imgs in enumerate(data_loader):
            
            imgs_pred = generate_predictions(model, imgs)
            write_preds_to_file(imgs_pred, batch_num, data_loader.batch_size, 
                image_file_names)


# Open predictions (saved as numpy) and scale up
def open_and_rescale_prediction(saved_prediction):
    """ Open epithelium predictions (10x magnification) and scale up to 
        same size as 40x magnification.

    Parameters
    ----------
    saved_prediction : npy filename
        Path to file with saved prediction as numpy array.

    Returns
    -------
    epi_mask : np array
        Numpy array containing epithelium prediction mask
    """

    epi_mask = np.load(saved_prediction)
    epi_mask = cv2.resize(epi_mask, (1136,1136), interpolation = cv2.INTER_NEAREST)

    return epi_mask


def get_epithelium_nuclei(json_file_name, epi_mask):
    """ Using epithelium mask and HoVerNet json file, identify epithelial
    nuclei.

    Parameters
    ----------
    json_file_name : str
        Path to HoVerNet json file
    epi_mask : np array
        Numpy array containing epithelium prediction mask

    Returns
    -------
    epi_nuc_uids : List
        List of nucleus IDs (int stored as str)
    epi_nuc_centroids: List
        List of centroid coordinates for each nucleus
    epi_nuc_contours : List
        List of contours for each nucleus
    """

    with open(json_file_name) as json_file:
        data = json.load(json_file)
    
    nuc_info = data['nuc']
    if not nuc_info:
        print(nuc_info)
        return None

    epi_nuc_uids = []
    epi_nuc_centroids = []
    epi_nuc_contours = []
    type_included = False
    
#    if 'type' in nuc_info["1"]:
    if 'type' in list(nuc_info.values())[0]:
        epi_nuc_types = []
        type_included = True

    for inst in nuc_info:

        inst_info = nuc_info[inst]
        centroid = np.around(inst_info['centroid']).astype(int)
        contour = inst_info['contour']
        if type_included:
            ntype = inst_info['type']
        
        if epi_mask[centroid[1], centroid[0]] == 255 and len(contour) >=5:
            epi_nuc_uids.append(inst)
            epi_nuc_centroids.append(centroid)
            epi_nuc_contours.append(contour)
            if type_included:
                epi_nuc_types.append(ntype)

    if type_included:
        return epi_nuc_uids, epi_nuc_centroids, epi_nuc_contours, epi_nuc_types

    return epi_nuc_uids, epi_nuc_centroids, epi_nuc_contours


def get_inst_map(mat_file_name):
    """Get nucleus instance map from HoVerNet mat file
    
    Parameters
    ----------
    mat_file_name : str
        Path to HoVerNet mat file

    Returns
    -------
    inst_map : np array
        Numpy array containing instance map
    """

    result_mat = sio.loadmat(mat_file_name)
    inst_map = result_mat['inst_map']

    return inst_map


def get_mean_h_concentrations(mat_file_name, image_file, epi_nuc_uids):
    """Get mean and max H concentrations for epithelial nuclei
    
    Parameters
    ----------
    mat_file_name : str
        Path to HoVerNet mat file
    image_file : str
        Path to 40x magnification image file
    epi_nuc_uids : List
        List of nucleus IDs

    Returns
    -------
    nuc_mean_h : List
        List of mean H concentration for each nucleus
    nuc_max_h : 
        List of max H concentration for each nucleus
    """

    inst_map = get_inst_map(mat_file_name)

    #image_file = os.path.join(hi_res_image_path, os.path.basename(image_file))
    #print(hi_res_image_file)

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    he_mat, he_conc = StainTransformer(normalise = True).extractor(image_file)

    image_dims = (2,) + inst_map.shape
    he_conc_sq = he_conc.reshape(image_dims)

    he_image = he_conc_sq[0, :, :]

    nuc_mean_h = []
    nuc_max_h = []

    for nuc_id in epi_nuc_uids:

        pix_indices = np.where(inst_map == int(nuc_id))
        pix_coords = list(zip(pix_indices[0], pix_indices[1]))

        mean_h = 0
        max_h = 0

        for row, col in pix_coords:
            mean_h = mean_h + he_image[row, col].item()
            max_h = max(max_h, he_image[row, col].item())

        nuc_mean_h.append(mean_h / len(pix_coords))
        nuc_max_h.append(max_h)

    return nuc_mean_h, nuc_max_h


def output_nuclei_stats(tile_id, epi_nuc_uids, epi_nuc_centroids, epi_nuc_contours, 
                        nuc_mean_h, nuc_max_h, epi_nuc_types=None):
    """Output data frame containing properties of epithelial nuclei, for one image tile.
    
    Parameters
    ----------
    tile_id : str
        Image tile basename
    epi_nuc_uids : List
        List of epithelial nucleus IDs
    epi_nuc_centroids : List
        List of epithelial nucleus centroid coords
    epi_nuc_contours : List
        List of epithilial nucleus contour coords
    nuc_mean_h : List
        List of nucleus mean H concentrations
    nuc_max_h : List
        List of nucleus max H concentrations
    epi_nuc_types : Boolean
        Is HoVerNet type available?

    Returns
    -------
    df : Pandas data frame
        Data frame of nucleus properties
    """
    
    ellipses = [cv2.fitEllipse(np.array(contour)) for contour in epi_nuc_contours]
    contour_areas = [cv2.contourArea(np.array(contour)) for contour in epi_nuc_contours]

    min_diams = [ellipse[1][0] for ellipse in ellipses]
    max_diams = [ellipse[1][1] for ellipse in ellipses]
    diam_ratios = [ellipse[1][0]/ellipse[1][1] for ellipse in ellipses] 
    ellipse_areas = [math.pi * 0.5 * max_diam * 0.5 * min_diam 
                        for max_diam, min_diam in zip(max_diams, min_diams)]

    df = pd.DataFrame(list(zip([tile_id] * len(epi_nuc_uids), epi_nuc_uids, 
                            epi_nuc_centroids, 
                            max_diams, min_diams, 
                            diam_ratios, ellipse_areas, contour_areas,
                            nuc_mean_h, nuc_max_h)), 
                        columns =['Tile ID', 'Nucl ID', 'Centroid', 'Min diam', 'Max diam', 
                                'Diam ratio', 'Ellipse area', 'Contour area', 'Mean H conc',
                                'Max H conc'])

    df['Type'] = epi_nuc_types
    
    return df


def run_model_for_predictions(model_path, image_file_names, batch_size, loader_workers):
    """Run a saved model to generate predictions from image files.
    Parameters
    ----------
    model_path : str
        Path to saved model
    image_file_names : List
        List of paths to image files
    batch_size : int
        Batch size to use
    loader_workers : int
        Number of loader workers to use

    """

    model = torch.load(model_path, map_location=DEVICE)
    model.eval()

    dataset = get_dataset(image_file_names)
    data_loader = get_data_loader(dataset, batch_size, loader_workers)
    
    output_predictions(model, data_loader, image_file_names)
    # No return statement as outputs predictions to files    


def save_sample_image(image_file, hov_path, tile_id, epi_mask, epi_nuc_contours):
    """Save a sample image showing original, epithlelium mask + epithelial nuclei
    Parameters
    ----------
    image_file : str
        Path to image
    hov_path : str
        Path to HoVerNet directory (containing json, mat, overlay dirs)
    tile_id : str
        Basename of image file
    epi_mask : np array
        Numpy array containing epithelium mask
    epi_nuc_contours : List
        List of contours for each nucleus
    
    """

    #hov_dir = os.path.abspath(os.path.join(json_file_path, os.pardir))
    overlay_file = os.path.join(hov_path, 'overlay', tile_id + '.png')
    overlay = cv2.imread(overlay_file)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (1136,1136), interpolation = cv2.INTER_NEAREST)

    ellipses = [cv2.fitEllipse(np.array(contour)) for contour in epi_nuc_contours]
    for ellipse in ellipses:
        image = cv2.ellipse(image, ellipse, (255,0,0), 2)
            
    fig, axes = plt.subplots(ncols = 3, sharey = True, figsize = (18,6))
            
    axes[0].imshow(overlay)  
    axes[1].imshow(epi_mask)
    axes[2].imshow(image)

    for ax in axes:
        ax.axis('off')

    plt.savefig(get_basename(image_file) + "epi_nuc.png")


def loop_through_tiles(image_file_names, hi_res_image_path, hov_path):
    """Run through input image files, generating predictions, identifying
    epithelial nuclei and saving results to file.

    Parameters
    ----------
    image_file_names : List 
        List of paths to image files
    hi_res_image_path : str
        Path to hi res (40x) image files
    hov_path : str
        Path to HoVerNet directory (containing json, mat, overlay dirs)

    Returns
    -------
    epi_nuc_data : Pandas data frame
        Data frame containing properties of all epithelial nuclei
    """

    epi_nuc_data = pd.DataFrame()

    hi_res_image_file_names = get_image_file_names(hi_res_image_path)
    rand_image_file = np.random.choice(hi_res_image_file_names)
    rand_image_tile_id = get_basename(rand_image_file)

    for image_file in image_file_names:

        #epi_mask = open_and_rescale_prediction('tmp/1182_16_15336_6816.npy')
        #json_file_name = '/home/ret58/rds/hpc-work/hover_net/consep_hi_res/json/1182_16_15336_6816.json'
        #tile_id = '1182_16_15336_6816'
        # Get tile ID
        # Get temp file name
        # Get json file name
        tile_id = get_basename(image_file)
        json_file_name = get_json_name(tile_id, hov_path)
        mat_file_name = get_mat_file_name(tile_id, hov_path)
        temp_file = os.path.join('tmp', tile_id + '.npy')

        if not os.path.exists(json_file_name):
            print(f"File {json_file_name} does not exist.")
            continue
        if not os.path.exists(mat_file_name):
            print(f"File {mat_file_name} does not exist.")
            continue
                 
        epi_mask = open_and_rescale_prediction(temp_file)

        epi_nuc_info = get_epithelium_nuclei(json_file_name, epi_mask)
        if epi_nuc_info is None:
            print(f"No nuclei identified in file {json_file_name}.")
            continue

        epi_nuc_uids, epi_nuc_centroids, epi_nuc_contours = epi_nuc_info[0:3]
        if len(epi_nuc_info) == 4:
            epi_nuc_types = epi_nuc_info[3]

        if len(epi_nuc_uids) == 0:
            print(f"No epithelial nuclei identified in file {json_file_name}.")
            continue

        hr_image_file = os.path.join(hi_res_image_path, tile_id.split('_')[0] + '_' + 
                                                        tile_id.split('_')[1], tile_id + '.png')
        if tile_id == '1182_16_10224_10224':
            print(hr_image_file)
        nuc_mean_h, nuc_max_h = get_mean_h_concentrations(mat_file_name, hr_image_file, epi_nuc_uids)

        df = output_nuclei_stats(tile_id, epi_nuc_uids, epi_nuc_centroids, 
                epi_nuc_contours, nuc_mean_h, nuc_max_h, epi_nuc_types)
        df.to_pickle('tmp/epi_nuc_' + tile_id + '.pkl')

        epi_nuc_data = pd.concat([epi_nuc_data, df], ignore_index = True)

        if tile_id == rand_image_tile_id:
            save_sample_image(rand_image_file, hov_path, tile_id, epi_mask, epi_nuc_contours)
      
    epi_nuc_data.to_pickle('epi_nuc_data.pkl')

    return epi_nuc_data


if __name__ == "__main__":

    command_line_args = parse_command_line_args()

    image_file_names = get_image_file_names(command_line_args.image_path)
    print(image_file_names[0])

    if command_line_args.predictions:
        run_model_for_predictions(command_line_args.model_path, image_file_names, 
            command_line_args.bs, command_line_args.lw)

    epi_nuc_data = loop_through_tiles(image_file_names, command_line_args.hi_res_image_path, 
                                    command_line_args.hov_path)

    print("Head and tail of final dataframe:")
    print(epi_nuc_data.head())
    print(epi_nuc_data.tail())
