import numpy as np
import torch
from torch.utils.data import Dataset

class ImagesDataset(Dataset):
    def __init__(self, images, labels, transform=None,label_as_strings=False,cut_around_center=False):
        # Define a mapping from string classes to numerical labels

        self.img = images
        self.labels = labels
        self.transform = transform

        class_to_label = {
            'AGN': 0.0,
            'SN': 1.0,
            'VS': 2.0,
            'asteroid': 3.0,
            'bogus': 4.0
        }
        if label_as_strings:
            print(labels)
            self.labels = torch.Tensor([class_to_label[c] for c in labels])
        
        if cut_around_center:
            self.img = self.img[:, :, 14:42, 14:42]

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]
        
        # Apply the transform if it's provided
        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]  # Adjusted indexing for features and labels
