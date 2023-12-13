import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


def train_net(net, features, labels, criterion=nn.MSELoss(), num_epochs=75, batch_size=256, test_size=0.2):
    # Split dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)

    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # Training
    for epoch in range(num_epochs):
        # Convert numpy arrays to PyTorch tensors
        for i in range(0, len(x_train), batch_size):
            x_batch = torch.from_numpy(x_train[i:i + batch_size]).float()
            y_batch = torch.from_numpy(y_train[i:i + batch_size]).float()

            # Forward pass
            outputs = net(x_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Convert test set to tensors
    x_test_tensor = torch.from_numpy(x_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    # Compute loss on the test set
    test_outputs = net(x_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)

    return net, test_loss.item()
