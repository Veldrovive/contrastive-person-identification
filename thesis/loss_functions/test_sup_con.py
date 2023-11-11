import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from thesis.loss_functions.sup_con import SupConLoss

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.normalize = nn.functional.normalize

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        features = self.fc5(x)
        features = self.normalize(features, dim=1)
        return features

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
validset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
validloader = DataLoader(validset, batch_size=64, shuffle=True)

# Instantiate the model, loss function, and optimizer
input_size = 28 * 28  # MNIST images are 28x28
hidden_size = 256
output_size = 3  # The size of the feature vector for the contrastive loss
model = MLP(input_size, hidden_size, output_size)
loss_function = SupConLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 50
while True:
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in trainloader:
            features = model(images)
            loss = loss_function(features, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(trainloader)}")

    # Plot the features
    features = []
    labels = []
    for images, labels_batch in trainloader:
        features_batch = model(images)
        features.append(features_batch.detach().numpy())
        labels.append(labels_batch.detach().numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
    ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels)
    plt.show()

    # Train a linear classifier on the train set features
    clf = LogisticRegression(random_state=0).fit(features, labels)
    print(f"Train Accuracy (Linear): {accuracy_score(labels, clf.predict(features))}")

    # Train a KNN classifier on the train set features
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(features, labels)
    print(f"Train Accuracy (KNN): {accuracy_score(labels, neigh.predict(features))}")

    # Evaluate the linear classifier on the validation set features
    features = []
    labels = []
    for images, labels_batch in validloader:
        features_batch = model(images)
        features.append(features_batch.detach().numpy())
        labels.append(labels_batch.detach().numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    print(f"Validation Accuracy (Linear): {accuracy_score(labels, clf.predict(features))}")

    # Evaluate the KNN classifier on the validation set features
    print(f"Validation Accuracy (KNN): {accuracy_score(labels, neigh.predict(features))}")

    # Plot the features
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
    ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels)
    plt.show()