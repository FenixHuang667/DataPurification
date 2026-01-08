import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os


#Key function: load_images
# import image data from MNIST-MIX, allowing filter for
# 1) digit as a vector.
# 2) language as a vector.
# 3) Number of data imported for specific language. The number of data is evely distributed to each digit.

######### Set up LetNet model for MNIST data
class LeNetEncoder(nn.Module):
    def __init__(self, output_dim=10):
        super(LeNetEncoder, self).__init__()
        """
        #Jingyi's setup
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 28x28
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 14x14


        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 120
        self.fc2 = nn.Linear(120, 84)  # 84
        self.fc3 = nn.Linear(84, output_dim)

        """
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2)  # 28x28
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5)  # 14x14

        self.fc1 = nn.Linear(8 * 5 * 5, 64)  # 120
        self.fc2 = nn.Linear(64, 32)  # 84
        self.fc3 = nn.Linear(32, output_dim)


        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # [B, 6, 14, 14]
        x = self.pool(self.relu(self.conv2(x)))  # [B, 16, 5, 5]
        x = x.view(x.size(0), -1)  # [B, 400]
        x = self.relu(self.fc1(x))  # [B, 120]
        x = self.relu(self.fc2(x))  # [B, 84]
        x = self.fc3(x)  # [B, output_dim]

        return x

class Classifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.fc3.out_features, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        output = self.classifier(features)
        return features, output


class LogisticRegressionImage(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        """
        Simple logistic regression for image data.

        Args:
            input_size: Number of input features (28*28=784 for MNIST)
            num_classes: Number of output classes
        """
        super(LogisticRegressionImage, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            features: Flattened input (batch_size, 784)
            logits: Output logits (batch_size, num_classes)
        """
        features = x.view(x.size(0), -1)
        logits = self.linear(features)
        return features, logits


######### Load raw MNIST-MIX data, with specific digits, language, and number of data
def load_images(
        digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        languages=['ARDIS', 'EMNIST'],
        samples_per_language=[100, 100],
        data_dir='./../../data/Image/MIX/',
        index_files=[None, None],
        verbose=False
):
    """
    Load images from multiple languages with specified sample counts.

    Args:
        digits: List of digits to load
        languages: List of language names
        samples_per_language: List of total sample counts per language
        data_dir: Directory containing .npz files
        index_files: List of index file names (one per language). If None, randomly sample and save indices.
        verbose: Print loading information

    Returns:
        images: numpy array of all images
        labels: numpy array of remapped labels (0-indexed)
    """

    # np.random.seed(42)

    all_images = []
    all_labels = []

    for language, total_samples, index_file in zip(languages, samples_per_language, index_files):
        npz_file = os.path.join(data_dir, f"{language}_train_test.npz")
        data = np.load(npz_file)

        combined_images = np.concatenate([data['X_train'], data['X_test']])
        combined_labels = np.concatenate([data['y_train'], data['y_test']])

        mask = np.isin(combined_labels, digits)
        combined_images = combined_images[mask]
        combined_labels = combined_labels[mask]

        available_samples = len(combined_images)
        if available_samples < total_samples:
            print(f"Warning: {language} only has {available_samples} samples, but {total_samples} requested")
            print(f"Using all {available_samples} available samples")
            total_samples = available_samples

        samples_per_digit = total_samples // len(digits)

        if index_file is None:
            sampled_indices = []
            for digit in digits:
                digit_mask = (combined_labels == digit)
                digit_indices = np.where(digit_mask)[0]

                if len(digit_indices) > samples_per_digit:
                    selected = np.random.permutation(len(digit_indices))[:samples_per_digit]
                    digit_indices = digit_indices[selected]

                sampled_indices.extend(digit_indices)

            sampled_indices = np.array(sampled_indices)

            index_filename = os.path.join(data_dir, f"{language}_indices_{total_samples}.npy")
            np.save(index_filename, sampled_indices)
        else:
            index_filename = os.path.join(data_dir, index_file)
            sampled_indices = np.load(index_filename)

        combined_images = combined_images[sampled_indices]
        combined_labels = combined_labels[sampled_indices]

        if verbose:
            print(f"\nLanguage: {language} (requested {total_samples} samples)")
            for digit in digits:
                count = np.sum(combined_labels == digit)
                print(f"  Digit {digit}: {count} samples")
            print(f"  Total: {len(combined_labels)} samples")

        all_images.append(combined_images)
        all_labels.append(combined_labels)

    all_images = np.concatenate(all_images)
    all_labels = np.concatenate(all_labels)

    label_mapping = {digit: idx for idx, digit in enumerate(digits)}
    mapped_labels = np.array([label_mapping[label] for label in all_labels])

    return all_images, mapped_labels


def load_images_from_npz(npz_file, digits=None, verbose=False):
    """
    Load all images from a single npz file and convert to numpy arrays.

    Args:
        npz_file: Path to npz file
        digits: List of digits to filter (if None, load all digits in file)
        verbose: Print loading information

    Returns:
        images: numpy array of all images
        labels: numpy array of labels
    """
    data = np.load(npz_file)

    images_list = []
    labels_list = []

    if 'X_train' in data and len(data['X_train']) > 0:
        images_list.append(data['X_train'])
        labels_list.append(data['y_train'])

    if 'X_test' in data and len(data['X_test']) > 0:
        images_list.append(data['X_test'])
        labels_list.append(data['y_test'])

    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    if digits is not None:
        mask = np.isin(labels, digits)
        images = images[mask]
        labels = labels[mask]

    unique_labels = np.unique(labels)
    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    labels = np.array([label_mapping[label] for label in labels])

    if verbose:
        print(f"Loaded from {npz_file}")
        if digits is not None:
            for digit in digits:
                count = np.sum(labels == label_mapping[digit])
                print(f"  Digit {digit}: {count} samples")
        print(f"  Total: {len(labels)} samples")

    return images, labels



######### Convert raw data to data loader
def create_data_loader(images, labels, batch_size=64, shuffle=False, mask=None):
    """
    Convert images and labels to a DataLoader.

    Args:
        images: numpy array of images
        labels: numpy array of labels
        batch_size: Batch size for dataloader
        shuffle: Whether to shuffle the data
        mask: optional boolean array to filter data (only include where mask[i] is True)

    Returns:
        data_loader: DataLoader containing the data
    """
    if mask is not None:
        images = images[mask]
        labels = labels[mask]

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
    dataset = SimpleDataset(images, labels, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

def extract_partition_dataloader_image(images, labels, partition_array, partition_index, batch_size=64, shuffle=False):
    """
    Extract data belonging to a specific partition and return a DataLoader.

    Args:
        images: numpy array of images
        labels: numpy array of labels
        partition_array: 1D array of integers of length N, where each entry indicates which partition that data point belongs to
        partition_index: The partition index to extract
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data

    Returns:
        DataLoader containing only data from the specified partition
    """
    mask = partition_array == partition_index

    images_partition = images[mask]
    labels_partition = labels[mask]

    dataloader = create_data_loader(images_partition, labels_partition, batch_size, shuffle)

    return dataloader


### Mislabel annotation, random
def mislabel_data(labels, mislabel_fraction=0.1):
    """
    Randomly mislabel a fraction of the data.

    Args:
        labels: numpy array of labels (0-indexed)
        mislabel_fraction: Fraction of data to mislabel (0.0 to 1.0)

    Returns:
        mislabeled: numpy array with some labels randomly changed
    """
    mislabeled = labels.copy()

    n_samples = len(labels)
    n_mislabel = int(n_samples * mislabel_fraction)

    # Randomly select indices to mislabel
    mislabel_indices = np.random.choice(n_samples, n_mislabel, replace=False)

    # Get number of classes
    num_classes = len(np.unique(labels))

    # For each selected index, assign a different random label
    for idx in mislabel_indices:
        original_label = labels[idx]
        possible_labels = [l for l in range(num_classes) if l != original_label]
        mislabeled[idx] = np.random.choice(possible_labels)

    return mislabeled

### Mislabel annotation, Specific
def mislabel_specific_indices(labels, data_per_digit, num_mislabel):
    """
    Mislabel data at specific indices: 95-99, 195-199, ..., 995-999 (total 50 mislabeled).

    Args:
        labels: numpy array of labels (0-indexed)

    Returns:
        mislabeled: numpy array with labels changed at specific indices
    """
    mislabeled = labels.copy()
    num_classes = len(np.unique(labels))

    mislabel_indices = []
    for i in range(0, 2):
        start = i * data_per_digit + data_per_digit - num_mislabel
        end = i * data_per_digit + data_per_digit
        mislabel_indices.extend(range(start, end))

    for idx in mislabel_indices:
        if idx < len(labels):
            original_label = labels[idx]
            possible_labels = [l for l in range(num_classes) if l != original_label]
            mislabeled[idx] = np.random.choice(possible_labels)

    return mislabeled

