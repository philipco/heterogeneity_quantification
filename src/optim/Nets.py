import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # One input feature, one output

    def forward(self, x):
        return self.linear(x).flatten()
