import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from sklearn.datasets import make_blobs
import random

class GAUSSIAN(data.Dataset):
    url = ""
    filename = ""
    file_md5 = ""

    def __init__(self, root, n_samples, size, channels, std=1,
                 transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.channels = channels

        self.targets = np.zeros((n_samples, 1)) 
        if type(size) is not tuple:
            size = (size, size)
        n_features = size[0] * size[1] * channels
        X, y = make_blobs(n_samples=n_samples, centers=1, n_features=n_features, cluster_std=std,
                  random_state=random.randint(0, 1000))
        self.data = X.reshape(n_samples, size[0], size[1], channels)

    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]
        img = np.array(img)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = (img - img.min())
        img =  (img * (255 // (img.max() - img.min()))).round()
        if self.channels == 3:
            img = Image.fromarray(img, mode="RGB")
        else:
            img = Image.fromarray(img.squeeze(2), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
