''' 
Script to import directory of WSIs and corresponding epithelium masks and to output these
as overlapping patches.
Patches with > 75% background pixels currently discarded.
TO DO: Patch size and stride also hard-coded. Make these a command-line option.
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import openslide
from PIL import Image

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu as otsu


def get_args(name = 'default', level_dim = 2, 
    WSI_dir="/home/rosie/epithelium_slides/WSIs",
    mask_dir="/home/rosie/epithelium_slides/export_openslide",
    output_dir="/home/rosie/epithelium_slides/output_images"):
    
    return int(level_dim), WSI_dir, mask_dir, output_dir
 

def get_image_and_mask_names(WSI_dir, mask_dir):
 
    files = sorted(os.listdir(WSI_dir))

    image_file_names = []
    mask_file_names = []

    for file in files:
        image_file_names.append(os.path.join(WSI_dir, file))
        mask_file_names.append(os.path.join(mask_dir, file[0:-4] + "_epithelium_mask.tif"))

    print(image_file_names)
    print(mask_file_names)

    return list(zip(image_file_names, mask_file_names))


def get_patch_origin_coords(image, patch_size, stride, level_dim):
    # Need to make sure taking the same dimensions from image and mask.
    
    # Image dimensions at highest resolution:
    width, height = image.level_dimensions[0]

    # Starting coords are in coords of full sized image
    # The -1 stops us from outputting patches that overlap with edge of image

    x_coords = [x * stride[0]  * 2**level_dim for x in range(0,(width // (stride[0] * 2**level_dim) - 1))]
    y_coords = [y * stride[1]  * 2**level_dim for y in range(0,(height // (stride[1] * 2**level_dim) - 1))]

    return x_coords, y_coords


def get_otsu_threshold(image):

    WSI_grayscale = np.asarray(
                image.read_region(
                    (0,0),
                    4,
                    image.level_dimensions[4],
                ).convert('L')
            )
    threshold = otsu(WSI_grayscale) / 255
    # 0 is black, 1 is white!!!

    return(threshold)    


def find_background_fraction(extracted_patch, threshold):
    
    extracted_patch_gray = rgb2gray(np.asarray(extracted_patch))

    extracted_patch_thresholded = extracted_patch_gray > threshold
    
    background_fraction = extracted_patch_thresholded.sum() / (patch_size[0] * patch_size[1])

    return background_fraction


def extract_patch(image, x, y, level_dim, patch_size = (224, 224)):

    extracted_patch = image.read_region((x,y), level_dim, patch_size, ).convert('RGB')
      
    return extracted_patch

def extract_mask(mask, x, y, level_dim, patch_size = (224, 224)):
   
    patch = mask.read_region((x,y), level_dim, patch_size, ).convert('L')
    patch_array = np.asarray(patch).astype('uint8')*255
    extracted_mask = Image.fromarray(patch_array)

    return extracted_mask

def save_patch(patch, output_dir, image_file_name):

    patch_file_name = os.path.join(output_dir, os.path.basename(image_file_name)[0:-4])
    patch_file_name = patch_file_name + "_" + str(x) + "_" + str(y) + ".png"

    patch.save(patch_file_name)
    #print("Saved to: " + patch_file_name)


if __name__ == "__main__":
    
    #WSI_dir="/mnt/c/Users/rosie/WSL/helpful_code/data-loader/WSIs"
    #mask_dir="/mnt/c/Users/rosie/WSL/helpful_code/data-loader/masks"
    #output_dir="/mnt/c/Users/rosie/WSL/output_images"
#    WSI_dir="/home/rosie/epithelium_slides/WSIs"
#    mask_dir="/home/rosie/epithelium_slides/export_openslide"
#    output_dir="/home/rosie/epithelium_slides/output_images"

    level_dim, WSI_dir, mask_dir, output_dir = get_args(*sys.argv)
    print(level_dim)
    print(WSI_dir)
    print(mask_dir)
    print(output_dir)

    file_names = get_image_and_mask_names(WSI_dir, mask_dir)
    tol = 0.75

    for image_file_name, mask_file_name in file_names:
        print(image_file_name)
        print(mask_file_name)
        image = openslide.OpenSlide(image_file_name)
        mask = openslide.OpenSlide(mask_file_name)
        patch_size = (int(1136/ 2**level_dim), int(1136/ 2**level_dim))
        stride = (int(patch_size[0] / 2), int(patch_size[1] / 2))
    
        #print(image.level_dimensions)
        #print(mask.level_dimensions)
 
        threshold =  get_otsu_threshold(image)
        print(threshold)

        x_coords, y_coords = get_patch_origin_coords(image, patch_size, stride, level_dim)
        #print(x_coords)
        #print(y_coords)

        for x in x_coords:
            for y in y_coords:
                #if y == 16576 and x == 16576:

                extracted_patch = extract_patch(image, x, y, level_dim, patch_size = patch_size)
                background_fraction = find_background_fraction(extracted_patch, threshold)

                if background_fraction <= tol:
                    save_patch(extracted_patch, output_dir, image_file_name)

                    extracted_mask = extract_mask(mask, x, y, level_dim, patch_size = patch_size)
                    save_patch(extracted_mask, output_dir, mask_file_name)
