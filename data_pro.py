import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class FashionDataset(Dataset):
    def __init__(self, data, transform=None):
        self.fashion_MNIST = list(data.values)
        self.transform = transform

        label = []
        image = []

        for i in self.fashion_MNIST:
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        self.images = np.asarray(image).reshape(-1, 28, 28, 1).astype('float32')

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)


def get_data():
    # train_csv = pd.read_csv('./data/fashion-mnist_train.csv')
    # test_csv = pd.read_csv('./data/fashion-mnist_test.csv')

    # train_set = FashionDataset(train_csv, transform=transforms.Compose([transforms.ToTensor()]))
    # test_set = FashionDataset(test_csv, transform=transforms.Compose([transforms.ToTensor()]))

    trans = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=trans)
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=trans)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)

    return train_loader, test_loader
