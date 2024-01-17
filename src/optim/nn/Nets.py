import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRegression(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # One input feature, one output

    def forward(self, x):
        return self.linear(x).flatten()

class LogisticRegression(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # One input feature, one output

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).flatten()


class CNN_CIFAR10(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
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
    def __init__(self, input_size: int, output_size: int):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjusted size due to MNIST image size (28x28 -> 7x7 after pooling)
        self.fc2 = nn.Linear(128, output_size)

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



