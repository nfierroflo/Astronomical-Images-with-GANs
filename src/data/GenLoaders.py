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

def get_training_loaders(save_dir='data/',batch_size=32,file_name="stamp_dataset_21_new.pkl",synthetic_SN=torch.tensor([]),label_as_strings=False,with_labels=False):
    #Carga de datos
    with open(save_dir + file_name, "rb") as f:
        data = pk.load(f)

    #Separacion de los datos
    Train_dict = data['Train']
    Validation_dict = data['Validation']

    #Images
    train_images = Train_dict['images']
    validation_images = Validation_dict['images']
    
    try:
        labels_train = Train_dict['class']
        labels_val = Validation_dict['class']

    except:
        labels_train = Train_dict['labels']
        labels_val = Validation_dict['labels']


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
        

    train_dataset = ImagesDataset(train_images_resize,labels_train,label_as_strings=label_as_strings)
    validation_dataset = ImagesDataset(validation_images_resize,labels_val,label_as_strings=label_as_strings)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader= DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)


    return train_loader, val_loader

def get_loaders(save_dir='data/',batch_size=32,file_name="stamp_dataset_21_new.pkl",synthetic_SN=torch.tensor([]),label_as_strings=False,with_labels=False,syn_in_train=0,syn_in_val=0,syn_in_test=0):
    #Carga de datos
    with open(save_dir + file_name, "rb") as f:
        data = pk.load(f)

    #Separacion de los datos
    Train_dict = data['Train']
    Validation_dict = data['Validation']
    Test_dict = data['Test']

    train_images = Train_dict['images']
    validation_images = Validation_dict['images']
    test_images = Test_dict['images']

    labels_train = Train_dict['labels']
    labels_val = Validation_dict['labels']
    labels_test = Test_dict['labels']

    #Convert the NumPy array to a PyTorch tensor
    train_images_tensor = torch.tensor(train_images.transpose(0, 3, 1, 2), dtype=torch.float32)
    validation_images_tensor = torch.tensor(validation_images.transpose(0, 3, 1, 2), dtype=torch.float32)
    test_images_tensor = torch.tensor(test_images.transpose(0, 3, 1, 2), dtype=torch.float32)


    # Add synthetic data to the datasets
    if synthetic_SN.shape[0] != 0:
        if label_as_strings:
            train_images_tensor = torch.cat((train_images_tensor, synthetic_SN[:syn_in_train]), dim=0)
            labels_train = np.concatenate((labels_train, np.array(syn_in_train*['SN'])))
            validation_images_tensor = torch.cat((validation_images_tensor, synthetic_SN[syn_in_train:syn_in_train + syn_in_val]), dim=0)
            labels_val = np.concatenate((labels_val, np.array(syn_in_val*['SN'])))
            test_images_tensor = torch.cat((test_images_tensor, synthetic_SN[syn_in_train + syn_in_val:syn_in_train + syn_in_val + syn_in_test]), dim=0)
            labels_test = np.concatenate((labels_test, np.array(syn_in_test*['SN'])))
        else:
            # For training, add syn_in_train synthetic samples
            train_images_tensor = torch.cat((train_images_tensor, synthetic_SN[:syn_in_train]), dim=0)
            labels_train = torch.cat((torch.Tensor(labels_train), torch.ones(syn_in_train)), dim=0)

            # For validation, add syn_in_val synthetic samples
            validation_images_tensor = torch.cat((validation_images_tensor, synthetic_SN[syn_in_train:syn_in_train + syn_in_val]), dim=0)
            labels_val = torch.cat((torch.Tensor(labels_val), torch.ones(syn_in_val)), dim=0)

            # For test, add syn_in_test synthetic samples
            test_images_tensor = torch.cat((test_images_tensor, synthetic_SN[syn_in_train + syn_in_val:syn_in_train + syn_in_val + syn_in_test]), dim=0)
            labels_test = torch.cat((torch.Tensor(labels_test), torch.ones(syn_in_test)), dim=0)

    # Create datasets and loaders
    train_dataset = ImagesDataset(train_images_tensor, labels_train, label_as_strings=label_as_strings)
    validation_dataset = ImagesDataset(validation_images_tensor, labels_val,label_as_strings=label_as_strings)
    test_dataset = ImagesDataset(test_images_tensor, labels_test, label_as_strings=label_as_strings)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_mixed_binarydatasets_loaders(save_dir='data/',batch_size=32,file_name="converted_binaryTrue_stamp_dataset_28.pkl",synthetic_SN=torch.tensor([]),syn_in_train=0,syn_in_val=0,syn_in_test=0):
    #Carga de datos
    with open(save_dir + file_name, "rb") as f:
        data = pk.load(f)

    #Separacion de los datos
    Train_dict = data['Train']
    Validation_dict = data['Validation']
    Test_dict = data['Test']

    train_images = Train_dict['images']
    validation_images = Validation_dict['images']
    test_images = Test_dict['images']

    labels_train = Train_dict['labels']
    labels_val = Validation_dict['labels']
    labels_test = Test_dict['labels']

    #Convert the NumPy array to a PyTorch tensor
    train_images_tensor = torch.tensor(train_images.transpose(0, 3, 1, 2), dtype=torch.float32)
    validation_images_tensor = torch.tensor(validation_images.transpose(0, 3, 1, 2), dtype=torch.float32)
    test_images_tensor = torch.tensor(test_images.transpose(0, 3, 1, 2), dtype=torch.float32)


    # Add synthetic data to the datasets
    if synthetic_SN.shape[0] != 0:
        # For training, add syn_in_train synthetic samples
        train_images_tensor = torch.cat((train_images_tensor, synthetic_SN[:syn_in_train]), dim=0)
        labels_train = torch.cat((torch.Tensor(labels_train), torch.ones(syn_in_train)), dim=0)

        # For validation, add syn_in_val synthetic samples
        validation_images_tensor = torch.cat((validation_images_tensor, synthetic_SN[syn_in_train:syn_in_train + syn_in_val]), dim=0)
        labels_val = torch.cat((torch.Tensor(labels_val), torch.ones(syn_in_val)), dim=0)

        # For test, add syn_in_test synthetic samples
        test_images_tensor = torch.cat((test_images_tensor, synthetic_SN[syn_in_train + syn_in_val:syn_in_train + syn_in_val + syn_in_test]), dim=0)
        labels_test = torch.cat((torch.Tensor(labels_test), torch.ones(syn_in_test)), dim=0)

    # Create datasets and loaders
    train_dataset = ImagesDataset(train_images_tensor, labels_train)
    validation_dataset = ImagesDataset(validation_images_tensor, labels_val)
    test_dataset = ImagesDataset(test_images_tensor, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader