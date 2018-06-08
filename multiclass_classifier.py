import platform
import io
import os
import time
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision
import torchvision.transforms as transforms

TRAIN_DATA_DIR = "C:/Users/Anya/Documents/ai4good/face-emotion-analysis/fer2013/Data_Images_Facial_Expressions/Training_Others_Small/"
VALIDATE_DATA_DIR = "C:/Users/Anya/Documents/ai4good/face-emotion-analysis/fer2013/Data_Images_Facial_Expressions/PrivateTest_Others/"
BATCH_SIZE = 35

TMP_DIR = "C:/Users/Anya/Documents/ai4good/face-emotion-analysis/tmp/"

IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48

NUM_CLASSES = 6 #6 for multiclass classifier (we are excluding disgust class)

# TODO:
#       implement weighted max / loss function? to account for uneven training class sizes
#       try saving model (pickle it), and in seperate file loading model and using it to classify one (or a few) images
#       evaluate using BEST model: currently not working
#       PROBLEM: sometimes loss goes to NAN if learning rate too high and/or too many epochs. Why? Ask JP. How to fix. Want to do validation from model at epoch with best loss

# transform = transforms.Compose([
#     transforms.Grayscale(),
#     transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
#     transforms.ToTensor()])

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()])

############################################################
#
# LOAD data
#
############################################################

#transform = transforms.ToTensor()
#train_sampler = sampler.WeightedRandomSampler(weights=[1.0/7]*7, num_samples=400*7, replacement=True)       # want to pick ~400 from each class to train
train_data = torchvision.datasets.ImageFolder(TRAIN_DATA_DIR, transform=transform)
#train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)


#validate_sampler = sampler.WeightedRandomSampler(weights=[1.0/NUM_CLASSES]*NUM_CLASSES, num_samples=100*7, replacement=True)    # want to pick ~100 from each class to validate
validate_data = torchvision.datasets.ImageFolder(VALIDATE_DATA_DIR, transform=transform)
#validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=BATCH_SIZE, sampler=validate_sampler, shuffle=False)
validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=BATCH_SIZE, shuffle=True)



############################################################
#
# PLOT 1st image
#
############################################################

# image0, target0 = train_data[0]
# print(image0)
# print(image0.shape)         #[1,48,48]
# print()
# #print(image0[0,:,:])
# #image0 = image0[0,:,:]
# image0 = image0.squeeze()
# print(image0.shape)         # [48,48]

# # image0Transposed = np.transpose(image0, (1,2,0))
# # print(image0Transposed.shape)
# # print(image0Transposed)

# # IMAGE_SIZE = 48*48

#  # plot first image
# _ = plt.imshow(image0, cmap='gray')
# plt.show()

print(train_data.classes)

############################################################
#
# DEFINE THE MODEL
#
############################################################

KERNEL_SIZE_CONV = 5
STRIDE = 1
PADDING = 2
KERNEL_SIZE_POOL = 2

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = 1,  # 1, because grayscale
                out_channels = 6,      # model chooses 16 filters
                kernel_size = KERNEL_SIZE_CONV,
                stride = STRIDE,
                padding = PADDING),
            nn.MaxPool2d(kernel_size = KERNEL_SIZE_POOL) # now image size is 48/2 = 24
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = 6, # convention: use powers of 2
                out_channels = 12,
                kernel_size = KERNEL_SIZE_CONV,
                stride = STRIDE,
                padding = PADDING),
            nn.MaxPool2d(kernel_size = KERNEL_SIZE_POOL) # now image size is 24/2 = 12
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels = 12,
                out_channels = 24,
                kernel_size = KERNEL_SIZE_CONV,
                stride = STRIDE,
                padding = PADDING),
            nn.MaxPool2d(kernel_size = KERNEL_SIZE_POOL)) # now image is 12/2 = 6

        self.block4 = nn.Sequential(
            nn.Linear(864, NUM_CLASSES)    # in=6*6*24=864, out=7 (7 possible emotions)
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = out.view(-1, 864)   # flatten for nn.Linear
        return self.block4(out)

model = cnn()

# DEFINE OPTIMIZER AND CRITERION

learning_rate = 0.03
print("learning rate = {}".format(learning_rate))

optimizer = optim.SGD(model.parameters(), lr=learning_rate) # Use Adam() or SGD()
criterion = nn.CrossEntropyLoss() # lost / cost function

############################################################
#
# Train the model
#
############################################################

model.train()

t0 = time.time()
total_loss = []
num_epochs = 20

best_epoch_loss = 10000 # arbitrary large number
best_epoch_num = 0
best_state_dict = None     # weights of model with best loss

elapsedTime = None

for epoch in range(num_epochs):

    train_loss = 0
    n_iter = 0

    for images, targets in train_loader:

        # Reset (zero) the gradient buffer
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss.append(loss)

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Statistics
        train_loss += loss
        n_iter += 1

    epoch_loss = train_loss/n_iter
    #print("loss={}, train_loss={}, n_iter={}, epoch_loss={}".format(loss, train_loss, n_iter, epoch_loss))
    
    elapsedTime = time.time() - t0

    print('Epoch: {}/{}, Loss: {:.4f}\tTime Elapsed:  {} minutes {:.4f} seconds'.format(
        epoch+1, num_epochs, epoch_loss,int(elapsedTime//60), elapsedTime % 60))

    if epoch == 0:
        best_epoch_loss = epoch_loss
        best_epoch_num = 0
        best_state_dict = model.state_dict()
        print("bestEpoch = {}".format(best_epoch_num + 1))
        torch.save(model.state_dict(), 'mytraining.pt')     #todo: remove. not best solution

    elif epoch_loss < best_epoch_loss:
        best_epoch_loss = epoch_loss
        best_epoch_num = epoch
        best_state_dict = model.state_dict()
        print("bestEpoch = {}".format(best_epoch_num + 1))
        torch.save(model.state_dict(), 'mytraining.pt')     #todo: remove. not best solution
    
print()
print("Total Time Elapsed: {} minutes {:.4f} seconds".format(int(elapsedTime//60), elapsedTime % 60))

# modelDict = {"model": None,
#             "state_dict": best_state_dict,
#             "trainingTime": elapsedTime}
bestModelDict = {"best_loss": best_epoch_loss,
            "state_dict": best_state_dict,
            "epoch_num": best_epoch_num}
pickle.dump(bestModelDict, open(os.path.join(TMP_DIR, "bestModelDict.p"), "wb"))

# to load:
# lastSavedModelDict = pickle.load(open(os.path.join(TMP_DIR, "modelDict.p"), "rb"))
# model = CNN()
# model.load_state_dict(torch.load('Trained_Model.pth'))

############################################################
#
# EVALUATE THE MODEL
#
############################################################

# use best model for evaluation
model = cnn()
# model.load_state_dict(best_state_dict)
model.load_state_dict(torch.load('mytraining.pt'))


model.eval()

total = 0
correct = 0

for i, (images, targets) in enumerate(validate_loader):

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, targets)
    _, predicted = torch.max(outputs.data, 1)

    # Statistics
    total += targets.size(0)
    correct += (predicted == targets).sum()

    if i < 3:
        print("total={}, correct = {}, predicted={}, target={}".format(total, correct, predicted, targets))

print(correct)  # tensor(10788)
print(total)    # 28708
print('Accuracy on the test set: {}%'.format(100 * correct/total))