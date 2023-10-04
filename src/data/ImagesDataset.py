from torch.utils.data import Dataset
import numpy as np

class StampDataset(Dataset):
    def __init__(self, images, labels, transform=None,label_as_strings=False,cut_around_center=False):
        # Define a mapping from string classes to numerical labels

        self.img = images
        self.labels = labels
        self.transform = transform

        class_to_label = {
            'AGN': 0,
            'SN': 1,
            'VS': 2,
            'asteroid': 3,
            'bogus': 4
        }
        if label_as_strings:
            self.labels = np.array([class_to_label[c] for c in labels])
        
        if cut_around_center:
            self.img = self.img[:, 14:42, 14:42, :]

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]
        
        # Apply the transform if it's provided
        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]  # Adjusted indexing for features and labels
