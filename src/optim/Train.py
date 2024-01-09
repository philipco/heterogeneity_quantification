import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

NB_EPOCH = 50


def train_neural_network(net, features, labels, criterion, lr=0.01, nb_epoch=NB_EPOCH, batch_size=256, test_size=0.2):
    """Create train/test and train a neural network."""
    # Split dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)

    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Training
    for epoch in range(nb_epoch):
        # Convert numpy arrays to PyTorch tensors
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size].float()

            net.zero_grad()

            # Forward pass
            outputs = net(x_batch).float()
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Compute loss on the test set
    test_outputs = net(x_test)
    c = nn.MSELoss(reduction='none')
    test_loss = criterion(test_outputs, y_test)
    atomic_test_losses = c(test_outputs, y_test) # TODO : keep and save them.

    return net, test_loss.item()#, atomic_test_losses
