from statistics import mean
import numpy as np
import torch
#from PIL import Image
import cv2
#import torchmetrics
#import sklearn.metrics as met

def calculate_dice(predictions, targets):
    ''' Calculate dice score for predictions. '''
    smooth = 1.
    
    pix_labels, _ = torch.max(predictions, dim=1)
    pix_targets, _ = torch.max(targets, dim=1)
    intersection = torch.eq(pix_labels,pix_targets).sum().item()
    dice = 2 * (intersection + smooth) / (len(torch.flatten(pix_labels)) + len(torch.flatten(pix_targets)) + smooth)
    
    return dice

def calculate_jaccard(predictions, targets):
    '''
    From https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    

    def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou
    '''



    smooth = 1.
    
    pix_labels, _ = torch.max(predictions, dim=1)
    pix_targets, _ = torch.max(targets, dim=1)
    intersection = torch.eq(pix_labels,pix_targets).sum().item()
    # Calculate union
    
    # Jaccard = mean (intersetion/(union + smooth) over classes
    return jaccard

pred_raw = np.load('/home/ret58/rds/hpc-work/epi_merge_consep/tmp/1266_15_19312_9088.npy')
pred = cv2.resize(pred_raw, dsize = (284, 284), interpolation = cv2.INTER_NEAREST)

mask = cv2.imread('/home/ret58/rds/hpc-work/epithelium_slides/masks/1266_15_epithelium_mask_19312_9088.png', 
                cv2.IMREAD_GRAYSCALE)

print(pred.shape)
print(mask.shape)

print(np.max(pred))
print(np.max(mask))

#mask_bin = mask / 255
#pred_bin = pred / 255

#print(mask_bin)
#print(pred_bin)

#jacc = met.jaccard_score(mask_bin, pred_bin)
#print(jacc)

pred = torch.from_numpy(pred)
mask = torch.from_numpy(mask)

pred = torch.unsqueeze(torch.unsqueeze(pred, dim = 0), dim = 0)
mask = torch.unsqueeze(torch.unsqueeze(mask, dim = 0), dim = 0)

print(pred[0,0,:,0])
print(mask[0,0,:,0])

dice = calculate_dice(pred, mask)

pred_noclasses, _ = torch.max(pred, dim=1)
mask_noclasses, _ = torch.max(mask, dim=1)

pred_noclasses[pred_noclasses == 255] = 1
mask_noclasses[mask_noclasses == 255] = 1

#jaccard = torchmetrics.JaccardIndex(num_classes=2)
#jacc = torchmetrics.functional.jaccard_index(pred_noclasses, mask_noclasses, num_classes = 2)


print(dice)
#print(jacc)