# Segmentation education

Rosie's internship project code: 
    segmentation of epithelium using a U-Net
    merging epithelium results with HoVerNet nucleus predictions

Note: Anything in a subfolder may need paths to modules updating before it will run!

# Main bits of code:

seg_UNet.py - Original experiments using the VOC segmentation experiement
Uses unet.py

seg_epi.py - Performs the epithelium segmentation.
Uses unet.py and image_dataset.py

epi_hover_merge.py - Merges epithelium predictions with HoVerNet results
Uses unet.py and image_dataset.py

plot_loss_acc.py - Plots chart of losses and accuracies for each epoch

run_db.bash - Script to output command line args for all runs in a single file

# images folder

Contains code used to extract patches from WSIs

# odds_and_ends folder

Contains bits and bobs used when exploring datasets, draft code etc.

# visualizations folder

Contains code for outputting images from trained models.


