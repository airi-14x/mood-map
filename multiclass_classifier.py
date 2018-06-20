import platform
import io
import os
import time
import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg") # use other backend for images?? 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision
import torchvision.transforms as transforms

from globalcontrast import GCNorm

TRAIN_DATA_DIR = "../Data_Images_Facial_Expressions/Training_Others/"
VALIDATE_DATA_DIR = "../Data_Images_Facial_Expressions/PrivateTest_Others/"
TMP_DIR = "tmp/"

NUM_CLASSES = 6 #(we are excluding disgust class)
BATCH_SIZE = 128

IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48

KERNEL_SIZE_CONV = 5
STRIDE = 1
PADDING = 2
KERNEL_SIZE_POOL = 2

#LEARNING_RATE = 0.01
LEARNING_RATE = 0.0001
NUM_EPOCHS=300

use_gpu = torch.cuda.is_available()
print("GPU Available: {}".format(use_gpu))

# TODO:
#       implement weighted max / loss function? to account for uneven training class sizes
#       try saving model (pickle it), and in seperate file loading model and using it to classify one (or a few) images
#       PROBLEM: sometimes loss goes to NAN if learning rate too high and/or too many epochs. Why? Ask JP. How to fix. Want to do validation from model at epoch with best loss

############################################################
#
# FUNCTION: LOAD DATA
#
############################################################

# transform = transforms.Compose([
#     transforms.Grayscale(),
#     transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
#     transforms.ToTensor()])

# transform = transforms.ToTensor()

def load_data():
    transform_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomAffine(degrees=(-15, 15), scale=(0.8, 1.2)),
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.RandomHorizontalFlip(p=0.5),
        GCNorm()
    ])

    transform_validate = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        GCNorm()
    ])

    train_data = torchvision.datasets.ImageFolder(TRAIN_DATA_DIR, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    classes = train_data.classes
    print("Train data classes: {}".format(train_data.classes))

    validate_data = torchvision.datasets.ImageFolder(VALIDATE_DATA_DIR, transform=transform_validate)
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

    return [train_loader, validate_loader, classes]

############################################################
#
# CLASS: DEFINE THE MODEL
#
############################################################

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = 1,  # 1, because grayscale
                out_channels = 32,      # model chooses 16 filters
                kernel_size = KERNEL_SIZE_CONV,
                stride = STRIDE,
                padding = PADDING),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True), #  NEW, standardize the weights
            nn.MaxPool2d(kernel_size = KERNEL_SIZE_POOL), # now image size is 48/2 = 24
            nn.Dropout(p=0.7)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, # convention: use powers of 2
                out_channels = 64,
                kernel_size = KERNEL_SIZE_CONV,
                stride = STRIDE,
                padding = PADDING),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size = KERNEL_SIZE_POOL), # now image size is 24/2 = 12
            nn.Dropout(p=0.7)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels = 64,
                out_channels = 128,
                kernel_size = KERNEL_SIZE_CONV,
                stride = STRIDE,
                padding = PADDING),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size = KERNEL_SIZE_POOL), # now image is 12/2 = 6
            nn.Dropout(p=0.7)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels = 128,
                out_channels = 256,
                kernel_size = KERNEL_SIZE_CONV,  # now image is 6/2 = 3
                stride = STRIDE,
                padding = PADDING))
            # No pooling layer this time

        self.block5 = nn.Sequential(
            nn.Linear(9216, 1000),
            nn.Linear(1000, NUM_CLASSES)    # in=6*6*256=9216, out=6 (6 possible emotions)
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out.view(-1, 9216)   # flatten for nn.Linear
        return self.block5(out)

############################################################
#
# FUNCTION: DEFINE OPTIMIZER AND CRITERION
#
############################################################

def optimizer_and_criterion(model):
    print("learning rate = {}".format(LEARNING_RATE))
    print("batch size = {}".format(BATCH_SIZE))

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-03) # Use Adam() or SGD()
    criterion = nn.CrossEntropyLoss() # lost / cost function

    return [optimizer, criterion]

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
        fig.savefig("tmp/graphs_final.png")
    else:
        fig.savefig("tmp/graphs_epoch_{}.png".format(epoch_num))
    if show_graph:
        plt.show()

############################################################
#
# FUNCTION: COMPUTE VALIDATION LOSS
#
############################################################

def compute_validation_loss(valModel, validate_loader, criterion):
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
        
        if use_gpu:
            images = images.cuda()
            targets = targets.cuda()

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
    accuracy = 100 * float(correct)/total
    # print('Accuracy on the validation set: {}%'.format(100 * correct/total))
    # print(classifications_matrix)
    # for i in range(NUM_CLASSES):
    #     print("Accuracy of {}'s: {}".format(i, classifications_matrix[i][i] /sum(classifications_matrix[i])))
    return [epoch_loss, accuracy]

############################################################
#
# FUNCTION: TRAIN THE MODEL
#
############################################################

def train_model(model, optimizer, criterion, 
    train_loader, validate_loader, training_losses=[], 
    validation_losses=[], training_accuracies=[], 
    validation_accuracies=[], num_epochs_trained=0):

    t0 = time.time()
    best_epoch_loss = 10000 # arbitrary large number
    best_epoch_num = 0
    best_state_dict = None     # weights of model with best loss
    elapsedTime = None

    model.train()

    for epoch in range(num_epochs_trained, NUM_EPOCHS):

        # if epoch == 50: # on 15th epoch, halve the learning rate
        #     learning_rate /= 2
        #     optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        #     print("***Reduced learning rate to {}***".format(learning_rate))

        train_loss = 0
        n_iter = 0
        total = 0
        correct = 0

        for images, targets in train_loader:
            

            if use_gpu:
                images = images.cuda()
                targets = targets.cuda()

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
        epoch_accuracy = 100 * float(correct)/total
        training_accuracies.append(epoch_accuracy)

        #torch.save(model.state_dict(), 'currentTrainingWeights.pt') 
        #del(outputs)

        # Compute validation metrics (loss, accuracy)
        val_loss, val_accuracy = compute_validation_loss(model, validate_loader, criterion)
        model.train()

        elapsedTime = time.time() - t0

        print('Epoch: {}/{}, Train Loss: {:.4f}\tVal Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Accuracy: {:.2f}\tTime Elapsed:  {} min {} s'.format(
            epoch+1, NUM_EPOCHS, epoch_loss, val_loss, epoch_accuracy, val_accuracy, int(elapsedTime//60), round(elapsedTime % 60)))
        training_losses.append(epoch_loss)

        validation_losses.append(val_loss)
        validation_accuracies.append(val_accuracy)

        if epoch == 0 or epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            best_epoch_num = epoch
            best_state_dict = model.state_dict()
            print("bestEpoch = {}".format(best_epoch_num + 1))
            # #save weights for model with best training loss. #todo: improve; not best solution
            torch.save(model.state_dict(), "tmp/bestTrainingWeights.pt")
        
        # Save learning curve and accuracy plots every 5 epochs (checkpoint)
        if (epoch + 1) % 5 == 0:
            make_plots(training_losses, validation_losses, training_accuracies, validation_accuracies, show_graph=False, epoch_num=epoch + 1)
            torch.save(model.state_dict(), "tmp/trainingWeights_epoch_{}.pt".format(epoch + 1))
            save_checkpoint(model, epoch, training_losses, validation_losses, training_accuracies, validation_accuracies, elapsedTime)


    print()
    print("Total Time Elapsed: {} minutes {:.4f} seconds".format(int(elapsedTime//60), elapsedTime % 60))

    bestModelDict = {"best_loss": best_epoch_loss,
                "state_dict": best_state_dict,
                "epoch_num": best_epoch_num}
    pickle.dump(bestModelDict, open(os.path.join(TMP_DIR, "bestModelDict.p"), "wb"))

    # to load:
    # lastSavedModelDict = pickle.load(open(os.path.join(TMP_DIR, "modelDict.p"), "rb"))
    # model = CNN()
    # model.load_state_dict(torch.load('Trained_Model.pth'))

    make_plots(training_losses, validation_losses, training_accuracies, validation_accuracies, show_graph=True)

    return model


############################################################
#
# FUNCTION: SAVE CHECKPOINT LOSSES, ACCURACIES AND WEIGHTS
#
############################################################

def save_checkpoint(model, epoch, training_losses, validation_losses,
    training_accuracies, validation_accuracies, training_time):
    checkpoint_dict = {
        'state_dict':model.state_dict(),
        'epoch': epoch,
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'training_accuracies': training_accuracies,
        'validation_accuracies': validation_accuracies,
        'training_time': training_time
    }

    pickle.dump(checkpoint_dict, open(os.path.join(TMP_DIR, "checkpoint_epoch_{}.pth".format(epoch + 1)), "wb"))

############################################################
#
# FUNCTION: EVALUATE THE MODEL
#
############################################################

def evaluate(model, validate_loader, criterion, classes):

    # use best model for evaluation
    model = cnn()
    model.load_state_dict(torch.load("tmp/bestTrainingWeights.pt"))

    classifications_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    classifications_matrix = classifications_matrix.astype(int)

    model.eval()

    total = 0
    correct = 0

    for i, (images, targets) in enumerate(validate_loader):
        
        if use_gpu:
            images = images.cuda()
            targets = targets.cuda()

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
        print("Accuracy of {}'s ({}): {}".format(i, classes[i], classifications_matrix[i][i] /sum(classifications_matrix[i])))

############################################################
#
# MAIN
#
############################################################


def main():

    # Load datasets
    train_loader, validate_loader, classes = load_data()

    # Define model
    #model = cnn()
    #from model import anne_model
    #model = anne_model()
    #from model2 import cnn
    #model = cnn()
    #from resnet_model import resnet18_mod
    #model = resnet18_mod()
    #from model2 import cnn_model
    #from model_JostineHo import cnn_model
    from best_cnn_model import cnn_model
    model = cnn_model()

    if use_gpu:
        model.cuda()

    # resume learning
    model.load_state_dict(torch.load("tmp/model3/trainingWeights_epoch_70.pt"))

    # Define optimizer and criterion
    optimizer, criterion = optimizer_and_criterion(model)

    checkpoint = pickle.load(open("tmp/model3/checkpoint_epoch_70.pth", "rb"))

    # Train the model and plot learning curve and accuracies
    # model = train_model(model, optimizer, criterion, train_loader, validate_loader)
    model = train_model(model, optimizer, criterion, train_loader, validate_loader, 
        checkpoint['training_losses'], checkpoint['validation_losses'], 
        checkpoint['training_accuracies'], checkpoint['validation_accuracies'], num_epochs_trained=70)


    # Evaluate the model
    evaluate(model, validate_loader, criterion, classes)


if __name__ == "__main__":
    main()
