import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # TODO
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
        self.fc1 = nn.Linear(in_features=256, out_features=120, bias=True)
        self.fc2 = nn.Linear(in_features=120, out_features=84, bias=True)
        self.fc3 = nn.Linear(in_features=84, out_features=10, bias=True)

    def forward(self, x):
        # TODO
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

    def name(self):
        return "ConvNet"

class Fully(nn.Module):
    def __init__(self):
        super(Fully, self).__init__()
        # TODO
        self.fc1 = nn.Linear(in_features=784, out_features=240, bias=True)
        self.fc2 = nn.Linear(in_features=240, out_features=10, bias=True)


    def forward(self, x):
        x = x.view(x.size(0),-1) # flatten input tensor
        # TODO
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

    def name(self):
        return "Fully"

