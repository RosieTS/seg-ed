
#%%
import torch
import torchvision
import torchvision.transforms as transforms

"""
Apart from U-net, all the below is taken from the tutorial below:
https://pytorch.org/tutorials/beginner/introyt/trainingyt.html 
"""
'''
I used the below to figure out that the target images are in "p" mode,
which means "palettised". Which is basically one channel with numbers
representing palette items, so I can just use that like a greyscale.

from PIL import Image
image = Image.open("data/VOCdevkit/VOC2012/SegmentationClass/2011_003271.png")
image.mode

The pixel classes are 1-20 for type of object, plus 0 for background
and 255 for void/unlabelled.
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html
'''

transform = transforms.Compose(
    [transforms.ToTensor(),
    # Resizing because images all different sizing. Could instead pad or make custom collate.
    transforms.Resize([500,500])])
    #, Jim said don't bother normalizing
    #transforms.Normalize((0.5,), (0.5,))])

import torchvision.transforms.functional as func

def convert_target(pil_img):
    grey = torch.squeeze(func.pil_to_tensor(pil_img))
    grey[grey == 255] = 21
    num_classes = 22
    # grey is a now tensor of shape (H, ground_truth = torch.eye(num_classes)[grey]W), 
    # and each pixel has an integer label
    ground_truth = torch.eye(num_classes)[grey.long()].permute(2, 0, 1)
    # Ground truth is now a tensor of shape (num_classes, H, W)
    # If grey[0, 0] = 0, ground_truth[:, 0, 0] = [1, 0, 0, ...]
    # if grey[0, 0] = 1, ground_truth[:, 0, 0] = [0, 1, 0, ...]
    return ground_truth.float()

transform_tgt = transforms.Compose([convert_target,
    transforms.Resize([500,500])])

# Set download = True if not yet downloaded/extracted.
training_set = torchvision.datasets.VOCSegmentation('data', 
    image_set = "train", download = False, transform = transform,
    target_transform = transform_tgt)

subsample = torch.randint(len(training_set), (64,))
training_set = torch.utils.data.Subset(training_set, subsample)

# Set download = True if not yet downloaded/extracted.
validation_set = torchvision.datasets.VOCSegmentation('data', 
    image_set = "val", download = False, transform = transform,
    target_transform = transform_tgt)

subsample = torch.randint(len(validation_set), (64,))
validation_set = torch.utils.data.Subset(validation_set, subsample)

#for image, ground_truth in training_set:
#    print(image.shape)
#    print(ground_truth.shape)
#    print("\n")
#exit()

batch_size = 1

training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True,
#    collate_fn = my_collate,
    num_workers=0) # changed from 2    
    #, Pinning memory apparently makes transfer to GPU faster 
#    pin_memory = True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False,
#    collate_fn = my_collate, 
    num_workers=0) # changed from 2
    #, Pinning memory apparently makes transfer to GPU faster 
#    pin_memory = True)

print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))


import matplotlib.pyplot as plt
import numpy as np

# Helper function for inline image display

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


dataiter = iter(training_loader)
images, labels = dataiter.next()

# Create a grid from the images and show them
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=False)
# Doesn't like the line below for some reason. Smth about not subsettable.
#print('  '.join(torch.classes[labels[j]] for j in range(4)))
##%%

#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#%%
# Telling the code to use the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Don't do this here - better to keep on CPU and load to GPU in batches
# https://stackoverflow.com/questions/69545355/when-to-put-pytorch-tensor-on-gpu
#for i, tr in enumerate(training_loader):
#    tr[0] = tr[0].to(device)
#    if i == 1: print("Training set device: ", tr[0].device)

#for i, va in enumerate(validation_loader):
#    va[0] = va[0].to(device)
#    if i == 1: print("Validation set device: ", va[0].device)

##%%
import unet

model = unet.UNet(num_classes = 22, num_layers = 2)
model.to(device)

#print(model)

#loss_fn = torch.nn.CrossEntropyLoss
loss_fn = torch.nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index):#, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Move data to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        #print("Device: ", inputs.device)
        #print("Data type: ", inputs.dtype)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        #outputs = model(inputs)
        # We need this to be softmax
        outputs = model(inputs).softmax(2)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        #if i % 1000 == 999:
        #if i % 10 == 9:
            #last_loss = running_loss / 1000 # loss per batch
        #last_loss = running_loss / 10 # loss per batch
        #print('  batch {} loss: {}'.format(i + 1, last_loss))
#           tb_x = epoch_index * len(training_loader) + i + 1
#           tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        #running_loss = 0.

        # Does deleting outputs free up memory?
        #del outputs

    last_loss = running_loss / (i + 1)

    return last_loss

# Switching off to make things as simple as possible.
#from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

import pynvml
from pynvml.smi import nvidia_smi
nvsmi = nvidia_smi.getInstance()
print(nvsmi.DeviceQuery('memory.free, memory.total'))

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Will this help with memory problems?
    # Can I run it between training and validation step?
    # torch.cuda.empty_cache()

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)#, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            #print(i)

            #nvsmi = nvidia_smi.getInstance()
            #print(nvsmi.DeviceQuery('memory.free, memory.total'))

            #torch.cuda.empty_cache()

            #nvsmi = nvidia_smi.getInstance()
            #print(nvsmi.DeviceQuery('memory.free, memory.total'))

            vinputs, vlabels = vdata

            # Move data to GPU
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)

            voutputs = model(vinputs).softmax(2)
            vloss = loss_fn(voutputs, vlabels)
            
            # running_vloss += vloss
            running_vloss += vloss.item()

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
#    writer.add_scalars('Training vs. Validation Loss',
#                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
#                    epoch_number + 1)
#    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1


# %%

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
# %%
# This cell gets some predictions from the saved model.

import torch
import torchvision
import unet

# Loading a model
#device = torch.device('cpu')
device = torch.device('cuda')
model = unet.UNet(num_classes = 22, num_layers = 2)
model.to(device)
# The map_Location=device doesn't seem to send weights to GPU.
#model.load_state_dict(torch.load(model_path, map_location=device))
model_path = 'model_20220531_114805_0'
model.load_state_dict(torch.load(model_path))

# evaluate model:
model.eval()

with torch.no_grad():
    # Picking 1 random batch. How to get just 1 image?
    img, lab = next(iter(validation_loader))

    img_gpu = img.to(device)
    output = model(img_gpu).softmax(2)
    print(output.shape)

    output = output.to('cpu')
    print(output.shape)
    print(output[:][:][0][0])

img_pred = np.argmax(output, axis = 1)

img_lab = np.argmax(lab, axis = 1)
img_lab = img_lab.type(torch.float)

#%%
# This displays my segmentation as a greyscale image.
# print(img_pred.shape)
# nping = torch.squeeze(img_pred).numpy()
# print(nping.shape)
# plt.imshow(nping, cmap="Greys")

#%%
# This cell takes the input, output and target and saves them all as images.

import torchvision.transforms.functional as func
from PIL import Image

def seg_to_pil(img_tensor):
    # Steal the palette from one of the images
    image = Image.open("data/VOCdevkit/VOC2012/SegmentationClass/2011_003271.png")
    #pixels = (np.array(image.getchannel(0)))
    pal = np.array(image.getpalette()).reshape(256,3)
    # Apply colour for 255 to value 21, rather than changing the value in the image.
    pal[21,:] = pal[255,:]
    pal = list(pal.flatten())

    pil_img = torchvision.transforms.ToPILImage()(img_tensor.type(torch.uint8))

    #print(np.array(pil_img))
    pil_img = pil_img.convert('P')
    pil_img.putpalette(pal)
    return pil_img

orig_img = func.to_pil_image(torch.squeeze(img))
orig_img.show()
orig_img.save("orig_img.png")

pil_pred = seg_to_pil(img_pred)
pil_pred.show()
pil_pred.save("pred_img.png")

pil_lab = seg_to_pil(img_lab)
pil_lab.show()
pil_lab.save("targ_img.png")

# %%
