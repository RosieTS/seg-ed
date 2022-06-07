## First attempt at script to run on Fawcett    ##
## Requires unet.py                             ##

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as func

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize([500,500])])

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
    image_set = "train", download = True, transform = transform,
    target_transform = transform_tgt)

#subsample = torch.randint(len(training_set), (64,))
#training_set = torch.utils.data.Subset(training_set, subsample)

# Set download = True if not yet downloaded/extracted.
validation_set = torchvision.datasets.VOCSegmentation('data', 
    image_set = "val", download = True, transform = transform,
    target_transform = transform_tgt)

#subsample = torch.randint(len(validation_set), (64,))
#validation_set = torch.utils.data.Subset(validation_set, subsample)

batch_size = 8

training_loader = torch.utils.data.DataLoader(training_set, 
    batch_size=batch_size, shuffle=True, num_workers=0)
validation_loader = torch.utils.data.DataLoader(validation_set, 
    batch_size=batch_size, shuffle=False, num_workers=0)
    
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))

# Telling the code to use the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

import unet

model = unet.UNet(num_classes = 22, num_layers = 2)
model.to(device)

loss_fn = torch.nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index):#, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        
        inputs, labels = data

        # Move data to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        #print("Device: ", inputs.device)
        #print("Data type: ", inputs.dtype)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        outputs = model(inputs).softmax(2)

        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    last_loss = running_loss / (i + 1)

    return last_loss


from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)#, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):

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

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

# Save some predictions

device = torch.device('cuda')
model = unet.UNet(num_classes = 22, num_layers = 2)
model.to(device)
model.load_state_dict(torch.load(model_path))

# evaluate model:
model.eval()

with torch.no_grad():
    # Picking 1 random batch. How to get just 1 image?
    img, lab = next(iter(validation_loader))

    img_gpu = img.to(device)
    output = model(img_gpu).softmax(2)
    #print(output.shape)

    output = output.to('cpu')
    #print(output.shape)
    #print(output[:][:][0][0])

# Just look at first image for now.
img = img[0]

img_pred = np.argmax(output[0], axis = 1)

img_lab = np.argmax(lab[0], axis = 1)
img_lab = img_lab.type(torch.float)

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
orig_img.save("orig_img.png")

pil_pred = seg_to_pil(img_pred)
pil_pred.save("pred_img.png")

pil_lab = seg_to_pil(img_lab)
pil_lab.save("targ_img.png")