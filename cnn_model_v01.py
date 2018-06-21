# This model uses batchnorm and dropout with p=0.7, to try to prevent overfitting.
# After training on a smaller dataset for 100 epochs, validation accuracy of ~37% was achieved
# (lr=0.2, batchsize=100, optimizer=SGD, criterion=crossEntropyLoss)
# TODO: Training for more epochs may improve accuracy.

import torch.nn as nn

NUM_CLASSES = 6
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

        # self.block4 = nn.Sequential(
        #     nn.Conv2d(in_channels = 128,
        #         out_channels = 256,
        #         kernel_size = KERNEL_SIZE_CONV,
        #         stride = STRIDE,
        #         padding = PADDING))
        #     # No pooling layer this time

        self.block5 = nn.Sequential(
            nn.Linear(4608, NUM_CLASSES)    # in=6*6*64=2304, out=6 (6 possible emotions)
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        # out = self.block4(out)
        out = out.view(-1, 4608)   # flatten for nn.Linear
        return self.block5(out)