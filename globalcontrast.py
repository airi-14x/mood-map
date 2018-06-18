import numpy as np
import torch

def global_contrast_normalization(img, s, lmda, epsilon):
    # img is already a numpy array
    X = img
    #print(X)

    # replacement for the loop
    X_average = np.mean(X)
    #print('Mean: ', X_average)
    X = X - X_average

    # `su` is here the mean, instead of the sum
    contrast = np.sqrt(lmda + np.mean(X**2))

    X = s * X / max(contrast, epsilon)
    #print(X)
    return torch.from_numpy(X).unsqueeze(0).float()

class GCNorm(object):
    def __call__(self, pic):
        return global_contrast_normalization(pic, 1, 10, 0.0000001)
    
