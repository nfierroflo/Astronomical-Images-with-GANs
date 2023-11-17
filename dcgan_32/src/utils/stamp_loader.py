import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pickle as pk
from torchvision import transforms
import torch.nn.functional as F

class StampDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.img = images[labels==1.0]
        self.labels = labels[labels==1.0]
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]
        
        # Apply the transform if it's provided
        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]  # Adjusted indexing for features and labels
    

def get_SN_dataset(save_dir, file_name="stamp_dataset_only_images_63.pkl", transform=None):
    
    with open(save_dir + file_name, "rb") as f:
        data = pk.load(f)

    #Training data
    train_dict, test_dict = data['Train'], data['Test']

    # Imgs and labels.
    train_images, test_images = train_dict['images'], test_dict['images']
    labels_train, labels_test = torch.Tensor(train_dict['class']=='SN'), torch.Tensor(test_dict['class']=='SN')

    # Convert the NumPy array to a PyTorch tensor
    train_images_tensor = torch.tensor(train_images.transpose(0, 3, 1, 2), dtype=torch.float32)
    test_images_tensor = torch.tensor(test_images.transpose(0, 3, 1, 2), dtype=torch.float32)
   
    # Loading Dataset
    train_dataset = StampDataset(train_images_tensor, labels_train, transform=transform)
    test_dataset = StampDataset(test_images_tensor, labels_test, transform=transform)

    return train_dataset, test_dataset   



class ImagesDataset(Dataset):
    def __init__(self, images, labels, transform=None,label_as_strings=False,cut_around_center=False, GPU=True):
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


def get_full_dataset(save_dir, file_name="stamp_dataset_only_images_63.pkl", transform=None, label_as_strings=True):
    
    with open(save_dir + file_name, "rb") as f:
        data = pk.load(f)

    #Training data
    train_dict, val_dict = data['Train'], data['Validation']

    # Imgs and labels.
    train_images, val_images = train_dict['images'], val_dict['images']
    labels_train, labels_val = train_dict['class'], val_dict['class']

    # Convert the NumPy array to a PyTorch tensor
    train_images_tensor = torch.tensor(train_images.transpose(0, 3, 1, 2), dtype=torch.float32)
    val_images_tensor = torch.tensor(val_images.transpose(0, 3, 1, 2), dtype=torch.float32)
   
    # Loading Dataset
    train_dataset = ImagesDataset(train_images_tensor, labels_train, transform=transform, label_as_strings=label_as_strings)
    test_dataset = ImagesDataset(val_images_tensor, labels_val, transform=transform, label_as_strings=label_as_strings)

    return train_dataset, test_dataset   

def get_training_loaders(save_dir, batch_size=32, file_name="stamp_dataset_28.pkl", 
                         synthetic_SN=torch.tensor([]), label_as_strings=False, with_labels=False,
                         desired_size=(28,28)):
    #Carga de datos
    with open(save_dir + file_name, "rb") as f:
        data = pk.load(f)

    #Separacion de los datos
    Train_dict = data['Train']
    Validation_dict = data['Validation']

    #Images
    train_images = Train_dict['images']
    validation_images = Validation_dict['images']

    labels_train = Train_dict['class']
    labels_val = Validation_dict['class']

    #numerical labels
    if with_labels:
        labels_train = Train_dict['labels']
        labels_val = Validation_dict['labels']


    #Convert the NumPy array to a PyTorch tensor
    train_images_tensor = torch.tensor(train_images.transpose(0, 3, 1, 2), dtype=torch.float32)
    validation_images_tensor = torch.tensor(validation_images.transpose(0, 3, 1, 2), dtype=torch.float32)
    
    train_images_resize = F.interpolate(train_images_tensor, size=desired_size, mode='bilinear', align_corners=False)
    validation_images_resize = F.interpolate(validation_images_tensor, size=desired_size, mode='bilinear', align_corners=False)

    if synthetic_SN.shape[0] != 0:
        train_images_resize = torch.cat((train_images_resize,synthetic_SN),dim=0)
        if label_as_strings:
            labels_train = torch.cat((labels_train, np.array(synthetic_SN.shape[0]*['SN'])))
        else:
            labels_train = torch.cat((torch.Tensor(labels_train), torch.ones(synthetic_SN.shape[0])),dim=0)
    
    else:
        train_images_resize = train_images_resize
        if label_as_strings:
            labels_train = torch.Tensor(labels_train)
        else:
            labels_train = torch.Tensor(labels_train)

    train_dataset = ImagesDataset(train_images_resize,labels_train,label_as_strings=label_as_strings)
    validation_dataset = ImagesDataset(validation_images_resize,labels_val,label_as_strings=label_as_strings)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader= DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)


    return train_loader, val_loader