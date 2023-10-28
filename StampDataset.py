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

def get_SNsLoader(save_dir,batch_size=32,file_name="stamp_dataset_21_new.pkl",label_as_strings=False,cut_around_center=False):
    
    with open(save_dir + file_name, "rb") as f:
        data = pk.load(f)

    #Separacion de los datos
    Train_dict = data['Train']
    #datos
    train_images = Train_dict['images']
    try:
        labels= torch.Tensor(Train_dict['class'])
    except:
        labels= torch.Tensor(Train_dict['labels'])
    class_to_label = {
        'AGN': 0.0,
        'SN': 1.0,
        'VS': 2.0,
        'asteroid': 3.0,
        'bogus': 4.0
    }

    if label_as_strings:
        labels_train = torch.Tensor([class_to_label[c] for c in labels])
    else:
        labels_train = torch.Tensor(labels)

    # Convert the NumPy array to a PyTorch tensor
    train_images_tensor = torch.tensor(train_images.transpose(0, 3, 1, 2), dtype=torch.float32)

    # Resize the images using torch.nn.functional.interpolate
    #desired_size = (28, 28)
    #train_images_resize=F.interpolate(train_images_tensor, size=desired_size, mode='bilinear', align_corners=False)

    if cut_around_center:
        train_images_resize = train_images_tensor[:, :, 14:42, 14:42]
   
    #Carga del dataset
    train_dataset = StampDataset(train_images_resize,labels_train)
    #train_dataset = StampDataset(np.transpose(train_images,[0,1,2,3]),features_train,labels_train,transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return dataloader   