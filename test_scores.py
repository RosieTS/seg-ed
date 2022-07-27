from statistics import mean
import numpy as np
import torch
#from PIL import Image
import cv2
#import torchmetrics
#import sklearn.metrics as met

def calculate_accuracy(predictions, targets):
    ''' Calculate pixel-wise accuracy of predictions. '''

    _, pix_labels = torch.max(predictions, dim=1)
    _, pix_targets = torch.max(targets, dim=1)
    correct = torch.eq(pix_labels,pix_targets).int()
    accuracy = correct.sum() / correct.numel()
    
    return accuracy

''' WRONG!!!!
def calculate_dice(predictions, targets):
    ''' Calculate dice score for predictions. '''
    smooth = 1.
    
    pix_labels, _ = torch.max(predictions, dim=1)
    pix_targets, _ = torch.max(targets, dim=1)
    intersection = torch.eq(pix_labels,pix_targets).sum().item()
    dice = 2 * (intersection + smooth) / (len(torch.flatten(pix_labels)) + len(torch.flatten(pix_targets)) + smooth)
    
    return dice
'''

def calculate_dice(predictions, targets):
    ''' Calculate dice score for predictions. '''
    smooth = 1.
     
    _, pix_labels = torch.max(predictions, dim=1)
    _, pix_targets = torch.max(targets, dim=1)
    # Operation will work only if background, mask have values 0, 1
    intersection = torch.sum(pix_labels*pix_targets)
    union = torch.sum(pix_labels) + torch.sum(pix_targets)
    dice = 2 * (intersection + smooth) / (union + smooth)
    
    return dice


def calculate_jaccard(predictions, targets):
    ''' Calculate dice score for predictions. '''
    smooth = 1.
     
    _, pix_labels = torch.max(predictions, dim=1)
    _, pix_targets = torch.max(targets, dim=1)
    # Operation will work only if background, mask have values 0, 1
    intersection = torch.sum(pix_labels*pix_targets)
    union = torch.sum(pix_labels) + torch.sum(pix_targets) - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    
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
acc = calculate_accuracy(pred, mask)

#pred_noclasses, _ = torch.max(pred, dim=1)
#mask_noclasses, _ = torch.max(mask, dim=1)

#pred_noclasses[pred_noclasses == 255] = 1
#mask_noclasses[mask_noclasses == 255] = 1

#jaccard = torchmetrics.JaccardIndex(num_classes=2)
#jacc = torchmetrics.functional.jaccard_index(pred_noclasses, mask_noclasses, num_classes = 2)


print(dice)
print(acc)
#print(jacc)