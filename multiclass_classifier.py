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
BATCH_SIZE = 10

TMP_DIR = "C:/Users/Anya/Documents/ai4good/face-emotion-analysis/tmp/"

IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48

NUM_CLASSES = 6 #(we are excluding disgust class)

# TODO:
#       implement weighted max / loss function? to account for uneven training class sizes
#       try saving model (pickle it), and in seperate file loading model and using it to classify one (or a few) images
#       PROBLEM: sometimes loss goes to NAN if learning rate too high and/or too many epochs. Why? Ask JP. How to fix. Want to do validation from model at epoch with best loss

############################################################
#
# LOAD data
#
############################################################

# transform = transforms.Compose([
#     transforms.Grayscale(),
#     transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
#     transforms.ToTensor()])

# transform = transforms.ToTensor()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()])

train_data = torchvision.datasets.ImageFolder(TRAIN_DATA_DIR, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
classes = train_data.classes
print("Train data classes: {}".format(train_data.classes))

validate_data = torchvision.datasets.ImageFolder(VALIDATE_DATA_DIR, transform=transform)
validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=BATCH_SIZE, shuffle=True)
print("Validate data classes: {}".format(validate_data.classes))

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
                out_channels = 32,      # model chooses 16 filters
                kernel_size = KERNEL_SIZE_CONV,
                stride = STRIDE,
                padding = PADDING),
            nn.MaxPool2d(kernel_size = KERNEL_SIZE_POOL) # now image size is 48/2 = 24
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, # convention: use powers of 2
                out_channels = 64,
                kernel_size = KERNEL_SIZE_CONV,
                stride = STRIDE,
                padding = PADDING),
            nn.MaxPool2d(kernel_size = KERNEL_SIZE_POOL) # now image size is 24/2 = 12
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels = 64,
                out_channels = 128,
                kernel_size = KERNEL_SIZE_CONV,
                stride = STRIDE,
                padding = PADDING),
            nn.MaxPool2d(kernel_size = KERNEL_SIZE_POOL)) # now image is 12/2 = 6

        # self.block4 = nn.Sequential(
        #     nn.Conv2d(in_channels = 128,
        #         out_channels = 256,
        #         kernel_size = KERNEL_SIZE_CONV,
        #         stride = STRIDE,
        #         padding = PADDING))
        #     # No pooling layer this time

        self.block5 = nn.Sequential(
            nn.Linear(4608, NUM_CLASSES)    # in=6*6*64=2304, out=7 (7 possible emotions)
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        # out = self.block4(out)
        out = out.view(-1, 4608)   # flatten for nn.Linear
        return self.block5(out)

model = cnn()

############################################################
#
# DEFINE OPTIMIZER AND CRITERION
#
############################################################

learning_rate = 0.02
print("learning rate = {}".format(learning_rate))

optimizer = optim.SGD(model.parameters(), lr=learning_rate) # Use Adam() or SGD()
criterion = nn.CrossEntropyLoss() # lost / cost function

############################################################
#
# FUNCTIONS: PLOT LEARNING CURVE
#
############################################################

def plot_learning_curve(training_losses, validation_losses):
    plt.title("Learning Curve (Loss vs time)")
    plt.ylabel("Loss")
    plt.xlabel("Training Steps (Epochs)")
    plt.plot(training_losses, label="training")
    plt.plot(validation_losses, label="validation")
    plt.legend(loc=1)

def plot_accuracies(training_accuracies, validation_accuracies):
    plt.title("Training and Validation Accuracies over time")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Steps (Epochs)")
    plt.plot(training_accuracies, label="training")
    plt.plot(validation_accuracies, label="validation")
    plt.legend(loc=1)

def make_plots(training_losses, validation_losses, training_accuracies, validation_accuracies, show_graph=False, epoch_num=None):
    #training_losses = [10,9,8,7,5.5,4,3,2,1]
    #validation_losses = [10,9,8,7,5.5,4.5,3.5,2.5,1.5]
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plot_learning_curve(training_losses, validation_losses)
    plt.subplot(1,2,2)
    plt.tight_layout(pad=1.1, w_pad=3.0, h_pad=3.0)
    plot_accuracies(training_accuracies, validation_accuracies)
    fig = plt.gcf()
    if epoch_num == None:
        fig.savefig("tmp/plot-checkpoint-final.png")
    else:
        fig.savefig("tmp/plot-checkpoint-{}-epochs.png".format(epoch_num))
    if show_graph:
        plt.show()

############################################################
#
# FUNCTION: COMPUTE VALIDATION LOSS
#
############################################################

def compute_validation_loss(valModel):
    #valModel = cnn()
    #valModel.load_state_dict(torch.load('currentTrainingWeights.pt'))
    valModel.eval()
    total = 0
    correct = 0

    # classifications_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    # classifications_matrix = classifications_matrix.astype(int)

    val_loss = 0
    n_iter = 0

    for i, (images, targets) in enumerate(validate_loader):

        # Forward pass
        outputs = valModel(images)
        loss = criterion(outputs, targets)

        _, predicted = torch.max(outputs.data, 1)

        # Statistics
        total += targets.size(0)
        correct += (predicted == targets).sum()
        val_loss += loss.data.item()

        # del(outputs)
        # del(loss)

        # for j in range(len(targets)):
        #     classifications_matrix[targets[j]][predicted[j]] += 1

        # if i == 0:
        #     print("First val batch: total={}, correct = {}, predicted={}, target={}".format(total, correct, predicted, targets))

        n_iter += 1

    epoch_loss = val_loss/n_iter
    accuracy = 100 * correct/total
    # print('Accuracy on the validation set: {}%'.format(100 * correct/total))
    # print(classifications_matrix)
    # for i in range(NUM_CLASSES):
    #     print("Accuracy of {}'s: {}".format(i, classifications_matrix[i][i] /sum(classifications_matrix[i])))
    return [epoch_loss, accuracy]


############################################################
#
# Train the model
#
############################################################

model.train()

t0 = time.time()
#total_loss = []
num_epochs = 100

training_losses = []
validation_losses = []

training_accuracies = []
validation_accuracies = []

best_epoch_loss = 10000 # arbitrary large number
best_epoch_num = 0
best_state_dict = None     # weights of model with best loss

elapsedTime = None

for epoch in range(num_epochs):

    # if epoch == 14: # on 15th epoch, lower the learning rate
    #     learning_rate = 0.01
    #     optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    #     print("***Reduced learning rate to {}***".format(learning_rate))

    train_loss = 0
    n_iter = 0
    total = 0
    correct = 0

    for images, targets in train_loader:

        # Reset (zero) the gradient buffer
        optimizer.zero_grad()

        # Forward pass (training)
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Statistics
        train_loss += loss.data.item()
        n_iter += 1
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum()

    epoch_loss = train_loss/n_iter
    #print("loss={}, train_loss={}, n_iter={}, epoch_loss={}".format(loss, train_loss, n_iter, epoch_loss))

    epoch_accuracy = 100 * correct/total
    training_accuracies.append(epoch_accuracy)

    #torch.save(model.state_dict(), 'currentTrainingWeights.pt') 
    #del(outputs)

    # Compute validation metrics (loss, accuracy)
    val_loss, val_accuracy = compute_validation_loss(model)
    model.train()

    elapsedTime = time.time() - t0

    print('Epoch: {}/{}, Train Loss: {:.4f}\tVal Loss: {:.4f}\tTime Elapsed:  {} minutes {:.4f} seconds'.format(
        epoch+1, num_epochs, epoch_loss, val_loss, int(elapsedTime//60), elapsedTime % 60))
    training_losses.append(epoch_loss)

    validation_losses.append(val_loss)
    validation_accuracies.append(val_accuracy)

    if epoch == 0 or epoch_loss < best_epoch_loss:
        best_epoch_loss = epoch_loss
        best_epoch_num = epoch
        best_state_dict = model.state_dict()
        print("bestEpoch = {}".format(best_epoch_num + 1))
        torch.save(model.state_dict(), 'bestTrainingWeights.pt')     #save weights for model with best training loss. #todo: improve; not best solution
    
    # Save learning curve and accuracy plots every 3 epochs (checkpoint)
    if (epoch + 1) % 1 == 0:
        make_plots(training_losses, validation_losses, training_accuracies, validation_accuracies, show_graph=False, epoch_num=epoch + 1)


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

make_plots(training_losses, validation_losses, training_accuracies, validation_accuracies, show_graph=True)


############################################################
#
# EVALUATE THE MODEL
#
############################################################

# use best model for evaluation
model = cnn()
# model.load_state_dict(best_state_dict)
model.load_state_dict(torch.load('bestTrainingWeights.pt'))

# classifications = {classID: [0]*NUM_CLASSES for classID in range(NUM_CLASSES)} # adjacency matrix to see which what misclassified labels are being missclassified as
# classifications_matrix [[0]*NUM_CLASSES]*NUM_CLASSES # adjacency matrix to see which what misclassified labels are being missclassified as
#                                                 # row i will contain what true labels i were classified as
#                                                 # columns contain all possible labels (0 .. NUM_CLASSES)
classifications_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
classifications_matrix = classifications_matrix.astype(int)

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

    for j in range(len(targets)):
        classifications_matrix[targets[j]][predicted[j]] += 1

    if i < 3:
        print("total={}, correct = {}, predicted={}, target={}".format(total, correct, predicted, targets))

print(correct)  # tensor(10788)
print(total)    # 28708
print('Accuracy on the validation set: {}%'.format(100 * correct/total))

print(classifications_matrix)
for i in range(NUM_CLASSES):
    print("Accuracy of {}'s: {}".format(i, classifications_matrix[i][i] /sum(classifications_matrix[i])))

print()

########################################################################################################
########################################################################################################
########################################################################################################
# TODO: Remove this section below

class_correct = list(0. for i in range(NUM_CLASSES))
class_total = list(0. for i in range(NUM_CLASSES))
with torch.no_grad():
    for data in validate_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(NUM_CLASSES):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))