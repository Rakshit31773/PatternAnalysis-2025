import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as transform
import numpy as np

IMG_EXT = ('.png', '.jpg', '.jpeg')
classes = {'NC' : 0, 'AD' : 1}

class ADNIDataset(Dataset):

    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.samples = []
        self.transform = transform
        self.target_transform = target_transform

        for c in classes.keys():
            sub = os.path.join(self.root_dir, c)
            if not os.path.isdir(sub):
                print(f"{sub} directory not found")
                continue

            for image in os.listdir(sub):
                if image.lower().endswith(IMG_EXT):
                    path = os.path.join(c, image)
                    self.samples.append((path, classes[c]))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")

        if(self.transform):
            image = self.transform(image)

        return image, label
