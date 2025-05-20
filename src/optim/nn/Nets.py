import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class LinearRegression(nn.Module):
    def __init__(self, input_size: int):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)#.flatten()


class LeNet(nn.Module):
    """From https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py.

    More example on: https://github.com/sushantkumar-estech/LeNet-CNN-for-CIFAR-10-Classification-Dataset/blob/master/LeNet_Convolutional_Neural_Network_for_CIFAR_10_Classification_Dataset.ipynb"""
    def __init__(self):
        super(LeNet, self).__init__()
        self.output_size = 10
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = F.avg_pool2d(out, 2)
        out = self.relu(self.conv2(out))
        out = F.avg_pool2d(out, 2)
        # out = out.view(out.size(0), -1)
        out = self.flatten(out)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class LogisticRegression(nn.Module):
    def __init__(self, input_size: int):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # One input feature, one output

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class HeartDiseaseRegression(LogisticRegression):
    def __init__(self):
        super(HeartDiseaseRegression, self).__init__(13)


class LiquidAssetRegression(nn.Module):
    def __init__(self):
        super(LiquidAssetRegression, self).__init__()
        self.l1 = nn.Linear(100, 64)
        self.l2 = nn.Linear(64, 32)
        self.last = nn.Linear(32, 1)

    def forward(self, x):
        out = F.elu(self.l1(x))
        out = F.elu(self.l2(out))
        return self.last(out)


class Synth2ClientsRegression(nn.Module):
    def __init__(self):
        super(Synth2ClientsRegression, self).__init__()
        self.linear = nn.Linear(2, 1, bias=True)

    def forward(self, x):
        return self.linear(x)#.flatten()


class Synth100ClientsRegression(nn.Module):
    def __init__(self):
        super(Synth100ClientsRegression, self).__init__()
        self.linear = nn.Linear(10, 1, bias=True)

    def forward(self, x):
        return self.linear(x)#.flatten()

class TcgaRegression(LinearRegression):
    def __init__(self):
        super(TcgaRegression, self).__init__(39)


class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        output_size = 10
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        output_size = 10
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjusted size due to MNIST image size (28x28 -> 7x7 after pooling)
        self.fc2 = nn.Linear(128, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x



