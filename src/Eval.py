import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset

import numpy as np

from PIL import Image

######### Evaluate the predictive accuracy of a model on a given test set (as data loader)

def evaluate_model(model, test_loader):
    """
    Evaluate model accuracy on test set.

    Args:
        model: Trained model
        test_loader: DataLoader for test data

    Returns:
        accuracy: Test accuracy (percentage)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features, outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total
    return accuracy

######### Compute the loss (average) of a model on a given test set (as data loader)

def compute_losses(model, test_loader):
    """
    Compute loss for each individual point and total loss.

    Args:
        model: Trained model from train_model
        test_loader: DataLoader containing the data

    Returns:
        individual_losses: numpy array of loss for each point
        total_loss: float, average loss over all points
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='none')
    individual_losses = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features, outputs = model(images)
            losses = criterion(outputs, labels)
            individual_losses.extend(losses.cpu().numpy())

    individual_losses = np.array(individual_losses)
    total_loss = np.mean(individual_losses)

    return individual_losses, total_loss

######### Run a train-test split to evaluate the heterogeneity of the data, for image data only
def split_train_test(images, labels, train_ratio=0.8, batch_size=64, shuffle=True):
    """
    Split data into train and test sets.

    Args:
        images: numpy array of images
        labels: numpy array of labels
        train_ratio: Ratio for training set
        batch_size: Batch size for dataloaders
        shuffle: Whether to shuffle before splitting

    Returns:
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
    """
    if shuffle:
        indices = np.random.permutation(len(labels))
        images = images[indices]
        labels = labels[indices]

    split_idx = int(train_ratio * len(labels))

    train_images = images[:split_idx]
    train_labels = labels[:split_idx]
    test_images = images[split_idx:]
    test_labels = labels[split_idx:]

    class SimpleDataset(Dataset):
        def __init__(self, images, labels, transform):
            self.images = images
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            image = self.images[idx]
            label = int(self.labels[idx])

            if image.ndim == 3:
                image = image.squeeze()

            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

            image = Image.fromarray(image)

            if self.transform:
                image = self.transform(image)

            return image, label

    transform = transforms.ToTensor()
    train_dataset = SimpleDataset(train_images, train_labels, transform)
    test_dataset = SimpleDataset(test_images, test_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# for voting data only
def split_train_test_vote(X, y, train_ratio=0.8, batch_size=64, shuffle=True):
    """
    Split voting data into train and test sets.

    Args:
        X: Feature tensor
        y: Label tensor
        train_ratio: Ratio for training set
        batch_size: Batch size for dataloaders
        shuffle: Whether to shuffle before splitting

    Returns:
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
    """
    if shuffle:
        indices = torch.randperm(len(y))
        X = X[indices]
        y = y[indices]

    split_idx = int(train_ratio * len(y))

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader




