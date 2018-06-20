import torch
import torch.nn as nn

KERNEL_SIZE_CONV = 5
STRIDE = 1
PADDING = 2
KERNEL_SIZE_POOL = 2
NUM_CLASSES = 6

# with RELUs

class cnn_model(nn.Module):
    def __init__(self):
        super(cnn_model, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = 1,  # 1, because grayscale
                out_channels = 32,      # model chooses 16 filters
                kernel_size = KERNEL_SIZE_CONV,
                stride = STRIDE,
                padding = PADDING),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True), #  NEW, standardize the weights
            nn.MaxPool2d(kernel_size = KERNEL_SIZE_POOL), # now image size is 48/2 = 24
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, # convention: use powers of 2
                out_channels = 64,
                kernel_size = KERNEL_SIZE_CONV,
                stride = STRIDE,
                padding = PADDING),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size = KERNEL_SIZE_POOL), # now image size is 24/2 = 12
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels = 64,
                out_channels = 128,
                kernel_size = KERNEL_SIZE_CONV,
                stride = STRIDE,
                padding = PADDING),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size = KERNEL_SIZE_POOL), # now image is 12/2 = 6
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        # self.block4 = nn.Sequential(
        #     nn.Conv2d(in_channels = 128,
        #         out_channels = 256,
        #         kernel_size = KERNEL_SIZE_CONV,  # now image is 6/2 = 3
        #         stride = STRIDE,
        #         padding = PADDING))
        #     # No pooling layer this time

        self.block5 = nn.Sequential(
            nn.Linear(128 * 6 * 6, 1000),
            nn.ReLU(),
            nn.Linear(1000, NUM_CLASSES)    # in=6*6*256=9216, out=6 (6 possible emotions)
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        #out = self.block4(out)
        out = out.view(-1, 128 * 6 * 6)   # flatten for nn.Linear
        return self.block5(out)