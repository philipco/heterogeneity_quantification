import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def train_neural_network(net, train_features, train_labels, criterion, nb_epochs, lr, batch_size):
    """Create train/test and train a neural network."""
    # Split dataset into train and test sets

    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Training
    print("=== Training the neural network. ===")
    for epoch in tqdm(range(nb_epochs)):
        # Convert numpy arrays to PyTorch tensors
        for i in range(0, len(train_features), batch_size):
            x_batch = train_features[i:i + batch_size]
            y_batch = train_labels[i:i + batch_size]

            net.zero_grad()

            # Forward pass
            outputs = net(x_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # We compute the train loss.
    train_loss = 0
    with torch.no_grad():
        for i in range(0, len(train_features), batch_size):
            x_batch = train_features[i:i + batch_size]
            y_batch = train_labels[i:i + batch_size]  # .float()

            # Forward pass
            outputs = net(x_batch).float()
            train_loss += criterion(outputs, y_batch)

    return net, train_loss / len(train_features) #, atomic_test_losses
