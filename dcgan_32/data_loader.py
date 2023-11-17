import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from src.utils.stamp_loader import get_SN_dataset
import os 

def get_data_loader(args):

    if args.dataset == 'cifar':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = dset.CIFAR10(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = dset.CIFAR10(root=args.dataroot, train=False, download=args.download, transform=trans)
    
    elif args.dataset == 'stamp':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset_directory = os.getcwd() + "/datasets/stamp/"
        train_dataset, test_dataset = get_SN_dataset(dataset_directory, transform=trans)


    elif args.dataset == 'stl10':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        train_dataset = dset.STL10(root=args.dataroot, split='train', download=args.download, transform=trans)
        test_dataset = dset.STL10(root=args.dataroot,  split='test', download=args.download, transform=trans)

    # Check if everything is ok with loading datasets
    assert train_dataset
    assert test_dataset

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader

