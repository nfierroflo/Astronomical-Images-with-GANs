import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle as pk
from torchvision import transforms
import torch.nn.functional as F
from src.data.StampDataset import StampDataset
from src.data.ImagesDataset import *

def get_SNsLoader(save_dir,batch_size=32,file_name="stamp_dataset_21_new.pkl"):
    
    with open(save_dir + file_name, "rb") as f:
        data = pk.load(f)

    #Separacion de los datos
    Train_dict = data['Train']
    #datos
    train_images = Train_dict['images']
    labels_train = torch.Tensor(Train_dict['class'])

    # Convert the NumPy array to a PyTorch tensor
    train_images_tensor = torch.tensor(train_images.transpose(0, 3, 1, 2), dtype=torch.float32)

    # Resize the images using torch.nn.functional.interpolate
    desired_size = (28, 28)

    train_images_resize=F.interpolate(train_images_tensor, size=desired_size, mode='bilinear', align_corners=False)
   
    #Carga del dataset
    train_dataset = StampDataset(train_images_resize,labels_train)
    #train_dataset = StampDataset(np.transpose(train_images,[0,1,2,3]),features_train,labels_train,transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return dataloader 

def get_training_loaders(save_dir='data/',batch_size=32,file_name="stamp_dataset_21_new.pkl",synthetic_SN=torch.tensor([]),label_as_strings=False):
    #Carga de datos
    with open(save_dir + file_name, "rb") as f:
        data = pk.load(f)

    #Separacion de los datos
    Train_dict = data['Train']
    Validation_dict = data['Validation']

    #Images
    train_images = Train_dict['images']
    validation_images = Validation_dict['images']

    #Labels
    labels_train = Train_dict['class']
    labels_val = Validation_dict['class']

    #Convert the NumPy array to a PyTorch tensor
    train_images_tensor = torch.tensor(train_images.transpose(0, 3, 1, 2), dtype=torch.float32)
    validation_images_tensor = torch.tensor(validation_images.transpose(0, 3, 1, 2), dtype=torch.float32)


    # Resize the images using torch.nn.functional.interpolate
    desired_size = (28, 28)
    
    train_images_resize = F.interpolate(train_images_tensor, size=desired_size, mode='bilinear', align_corners=False)
    validation_images_resize = F.interpolate(validation_images_tensor, size=desired_size, mode='bilinear', align_corners=False)

    if synthetic_SN.shape[0] != 0:
        train_images_resize = torch.cat((train_images_resize,synthetic_SN),dim=0)
        if label_as_strings:
            labels_train = np.concatenate((labels_train, np.array(synthetic_SN.shape[0]*['SN'])))
        else:
            labels_train = torch.cat((torch.Tensor(labels_train),torch.ones(synthetic_SN.shape[0])),dim=0)
        

    train_dataset = ImagesDataset(train_images_resize,labels_train,label_as_strings=True)
    validation_dataset = ImagesDataset(validation_images_resize,labels_val,label_as_strings=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader= DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)


    return train_loader, val_loader