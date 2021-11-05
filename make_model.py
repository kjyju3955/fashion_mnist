import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torchvision.models.resnet import ResNet, BasicBlock


class MNISTResNet(ResNet):
    def __init__(self):
        super(MNISTResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=3, bias=False)


'''
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()

        self.layer1 = nn.Sequential(
            (16,1,256,256) -->
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  (16,32,256,256)
            nn.BatchNorm2d(32), (16,32,256,256)
            nn.ReLU(), (16,32,256,256)
            nn.MaxPool2d(kernel_size=2, stride=2) (16,32,128,128)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=200)
        self.fc3 = nn.Linear(in_features=200, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
'''


def make_model(device):
    model = MNISTResNet()
    model.to(device)

    error = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(model)

    return model, error, optimizer
