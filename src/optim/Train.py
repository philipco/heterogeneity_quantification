import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split




def train_neural_network(net, train_features, train_labels, criterion, nb_epochs, lr, batch_size):
    """Create train/test and train a neural network."""
    # Split dataset into train and test sets

    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Training
    for epoch in range(nb_epochs):
        # Convert numpy arrays to PyTorch tensors
        for i in range(0, len(train_features), batch_size):
            x_batch = train_features[i:i + batch_size]
            y_batch = train_labels[i:i + batch_size]#.float()

            net.zero_grad()

            # Forward pass
            outputs = net(x_batch).float()
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    return net #, atomic_test_losses
