import torch

from torchvision import datasets, transforms
from . import BaseDataset


class ImageNetDataset(BaseDataset):
    def __init__(self, root='/home/share/dataset/text-to-images/imagenet', img_size=256):
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])
        self.dataset = datasets.ImageNet(root=root, split='train', transform=self.transform)

    def get_dataloader(self, batch_size):
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)


class CIFAR10Dataset(BaseDataset):
    def __init__(self, root='/moai/data/cifar10', img_size=64):
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])
        self.dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=self.transform)

    def get_dataloader(self, batch_size):
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

class CIFAR100Dataset(BaseDataset):
    def __init__(self, root='/moai/data/cifar100', img_size=64):
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])
        self.dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=self.transform)

    def get_dataloader(self, batch_size):
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)