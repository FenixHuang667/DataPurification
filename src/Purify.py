import torch
import numpy as np

from src.Variance import compute_potential
from src.LoadSyn import extract_partition_dataloader, create_dataloader
from src.Train import train_logistic_regression, train_model_image

from src.Eval import evaluate_model
from src.LoadImage import create_data_loader

def find_top_removal_potential(X, y, partition_array, models, k, t, batch_size=64, device=None):
    """
    Find the top t data points to remove from partition k that maximize potential decrease.

    Args:
        X: Feature tensor of shape (N, num_features)
        y: Label tensor of shape (N,)
        partition_array: 1D array of integers indicating partition membership
        models: List of models where models[i] is the model for partition i
        k: Working partition index
        t: Number of top points to return
        batch_size: Batch size for DataLoader
        device: Device to use

    Returns:
        top_original_indices: List of t indices in original dataset
        top_updated_models: List of t updated models after each removal
        top_potentials: List of t potential values after each removal
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = extract_partition_dataloader(X, y, partition_array, k, batch_size, shuffle=False)

    model = models[k]

    original_potential, updated_models, n = compute_potential(model, dataloader)

    # print(f"Dim: {n}, Original potential: {original_potential:.6f}")

    mask_partition = partition_array == k
    original_indices = np.where(mask_partition)[0]

    X_partition = X[mask_partition]
    y_partition = y[mask_partition]

    potential_decreases = []
    potentials = []

    for i in range(n):
        updated_model = updated_models[i]

        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_reduced = X_partition[mask]
        y_reduced = y_partition[mask]

        dataloader_reduced = create_dataloader(X_reduced, y_reduced, batch_size, shuffle=False)

        potential_i, _, n_i = compute_potential(updated_model, dataloader_reduced)

        decrease = original_potential - potential_i
        potential_decreases.append(decrease)
        potentials.append(potential_i)

    potential_decreases = np.array(potential_decreases)
    sorted_indices = np.argsort(potential_decreases)[::-1]
    top_t_indices = sorted_indices[:t]

    top_original_indices = [original_indices[i] for i in top_t_indices]

    if t == 1:
        top_updated_models = [updated_models[top_t_indices[0]]]
        top_potentials = [potentials[top_t_indices[0]]]
    else:
        top_updated_models = None
        top_potentials = None

    return top_original_indices, top_updated_models, original_potential


def find_top_removal_influence(X, y, partition_array, models, k, t, batch_size=64, device=None):
    """
    Find top t data points in partition k with highest I(z_i, z_i) values.

    Args:
        X: Feature tensor of shape (N, num_features)
        y: Label tensor of shape (N,)
        partition_array: 1D array of integers indicating partition membership
        models: List of models where models[i] is the model for partition i
        k: Working partition index
        t: Number of top points to return
        batch_size: Batch size for DataLoader
        device: Device to use

    Returns:
        top_indices: List of t indices in original dataset with highest diagonal influence
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from src.Variance import compute_influence_matrix

    dataloader = extract_partition_dataloader(X, y, partition_array, k, batch_size, shuffle=False)

    model = models[k]


    influence_matrix, _, _ = compute_influence_matrix(model, dataloader)
    influence_matrix = influence_matrix.to(device)

    diagonal_influence = -torch.diag(influence_matrix)

    sorted_indices = torch.argsort(diagonal_influence, descending=True)
    top_t_indices = sorted_indices[:t].cpu().numpy()

    mask_partition = partition_array == k
    original_indices = np.where(mask_partition)[0]

    top_original_indices = original_indices[top_t_indices]

    return top_original_indices.tolist()


def iterative_removal_partitions(X, y, partition_array, models, max_iterations, outlier_indices=None, batch_size=64,
                                 device=None):
    """
    Iteratively remove data points from each partition based on potential decrease.

    Args:
        X: Feature tensor of shape (N, num_features)
        y: Label tensor of shape (N,)
        partition_array: 1D array of integers indicating partition membership
        models: List of models where models[i] is the model for partition i
        max_iterations: Maximum number of removals per partition
        outlier_indices: List/array of outlier indices (if None, no outlier checking)
        batch_size: Batch size for DataLoader
        device: Device to use

    Returns:
        partition_array: Updated partition array with removed points marked as 0
        removal_records: List of tuples (partition_k, iteration, removed_idx, is_outlier)
    """
    partition_array = partition_array.copy()
    removal_records = []

    active_partitions = np.unique(partition_array[partition_array > 0])

    for k in active_partitions:
        for iteration in range(max_iterations):
            if np.sum(partition_array == k) < 10:
                break

            top_indices, top_models, top_potentials = find_top_removal_potential(
                X, y, partition_array, models, k, t=1, batch_size=batch_size, device=device
            )

            removed_idx = top_indices[0]

            if outlier_indices is not None:
                is_outlier = int(removed_idx in outlier_indices)
            else:
                is_outlier = None

            partition_array[removed_idx] = 0

            removal_records.append((k, iteration, removed_idx, is_outlier))

            print(f"Partition: {k}, Iteration: {iteration}, Removed Index: {removed_idx}, Potential: {top_potentials[0]:.4f}, Is Outlier: {is_outlier}")

            if top_models is not None:
                models[k] = top_models[0]

    return partition_array, removal_records




def Removal_process_vote(X, y, partition_array, models, max_iterations,
                                 test_loader = None, removal_method='top_p', t=10,
                                 input_size=None, num_classes=None, target_acc=0.99, lr=0.001,
                                 max_epochs=1000, verbose=False,
                                 outlier_indices=None, batch_size=64, device=None):
    """
    Iteratively remove data points from each partition based on different methods.

    Args:
        X: Feature tensor of shape (N, num_features)
        y: Label tensor of shape (N,)
        partition_array: 1D array of integers indicating partition membership
        models: List of models where models[i] is the model for partition i
        max_iterations: Maximum number of iterations per partition
        removal_method: 'one_by_one', 'top_t_potential', or 'top_t_influence'
        t: Number of points to remove per iteration (for top_t methods)
        input_size: Input size for model retraining
        num_classes: Number of classes for model retraining
        target_acc: Target accuracy for model retraining
        lr: Learning rate for model retraining
        max_epochs: Maximum epochs for model retraining
        verbose: Verbose flag for model retraining
        outlier_indices: List/array of outlier indices (if None, no outlier checking)
        batch_size: Batch size for DataLoader
        device: Device to use

    Returns:
        partition_array: Updated partition array with removed points marked as 0
        removal_records: List of tuples (partition_k, iteration, removed_idx, is_outlier)
    """
    partition_array = partition_array.copy()
    removal_records = []

    active_partitions = np.unique(partition_array[partition_array > 0])

    for k in active_partitions:
        for iteration in range(max_iterations):
            if np.sum(partition_array == k) < 10:
                break

            if test_loader is not None:
                acc = evaluate_model(models[k], test_loader)
                # print(f"Model {k}: Accuracy: {acc:.4f}")

            if removal_method == 'seq_p':
                top_indices, top_models, original_potentials = find_top_removal_potential(
                    X, y, partition_array, models, k, t=1, batch_size=batch_size, device=device
                )

                removed_idx = top_indices[0]

                if outlier_indices is not None:
                    is_outlier = int(removed_idx in outlier_indices)
                else:
                    is_outlier = None

                partition_array[removed_idx] = 0
                removal_records.append((k, iteration, removed_idx, is_outlier))

                if outlier_indices is not None:
                    print(
                        f"Partition: {k}, Iteration: {iteration}, Removed Index: {removed_idx}, Is Outlier: {is_outlier}")

                if top_models is not None:
                    models[k] = top_models[0]

            elif removal_method == 'top_p':
                top_indices, _, original_potential = find_top_removal_potential(
                    X, y, partition_array, models, k, t=t, batch_size=batch_size, device=device
                )

                n = np.sum(partition_array != 0)
                print(f"{iteration},{n},{original_potential:.6f},{acc:.4f}")

                for removed_idx in top_indices:
                    if outlier_indices is not None:
                        is_outlier = int(removed_idx in outlier_indices)
                    else:
                        is_outlier = None

                    partition_array[removed_idx] = 0
                    removal_records.append((k, iteration, removed_idx, is_outlier))

                    if outlier_indices is not None:
                        print(
                            f"Partition: {k}, Iteration: {iteration}, Removed Index: {removed_idx}, Is Outlier: {is_outlier}")

                dataloader = extract_partition_dataloader(X, y, partition_array, k, batch_size, shuffle=False)
                model, _ = train_logistic_regression(dataloader, input_size, num_classes)
                models[k] = model

            elif removal_method == 'top_i':

                dataloader = extract_partition_dataloader(X, y, partition_array, k, batch_size, shuffle=False)
                model, _ = train_logistic_regression(dataloader, input_size, num_classes)

                original_potential, _, n = compute_potential(model, dataloader)

                # print(f"Dim: {n}, Original potential: {original_potential:.6f}")
                print(f"{iteration},{n},{original_potential:.6f},{acc:.4f}")

                top_indices = find_top_removal_influence(
                    X, y, partition_array, models, k, t=t, batch_size=batch_size, device=device
                )

                for removed_idx in top_indices:
                    if outlier_indices is not None:
                        is_outlier = int(removed_idx in outlier_indices)
                    else:
                        is_outlier = None

                    partition_array[removed_idx] = 0
                    removal_records.append((k, iteration, removed_idx, is_outlier))

                    if outlier_indices is not None:
                        print(
                            f"Partition: {k}, Iteration: {iteration}, Removed Index: {removed_idx}, Is Outlier: {is_outlier}")

                dataloader = extract_partition_dataloader(X, y, partition_array, k, batch_size, shuffle=False)
                model, _ = train_logistic_regression(dataloader, input_size, num_classes)
                models[k] = model

            elif removal_method == 'rand':
                mask_partition = partition_array == k
                indices_k = np.where(mask_partition)[0]

                n_available = len(indices_k)

                if t > n_available:
                    continue

                selected_indices = np.random.choice(indices_k, t, replace=False)

                dataloader = extract_partition_dataloader(X, y, partition_array, k, batch_size, shuffle=False)

                model = models[k]

                original_potential, _, n = compute_potential(model, dataloader)

                # print(f"Dim: {n}, Original potential: {original_potential:.6f}")
                print(f"{iteration},{n},{original_potential:.6f},{acc:.4f}")

                for removed_idx in selected_indices:
                    if outlier_indices is not None:
                        is_outlier = int(removed_idx in outlier_indices)
                    else:
                        is_outlier = None

                    partition_array[removed_idx] = 0
                    removal_records.append((k, iteration, removed_idx, is_outlier))

                    if outlier_indices is not None:
                        print(
                            f"Partition: {k}, Iteration: {iteration}, Removed Index: {removed_idx}, Is Outlier: {is_outlier}")

                dataloader = extract_partition_dataloader(X, y, partition_array, k, batch_size, shuffle=False)
                model, _ = train_logistic_regression(dataloader, input_size, num_classes)
                models[k] = model

            else:
                raise ValueError("removal_method must be 'one_by_one', 'top_t_potential', or 'top_t_influence'")

    return partition_array, removal_records


def reassign_between_partitions(X, y, partition_array, i, j, models):
    """
    Reassign data between two partitions based on cross-model predictions.

    Args:
        X: Feature tensor of shape (N, num_features)
        y: Label tensor of shape (N,)
        partition_array: 1D array of integers indicating partition membership
        i: First partition index
        j: Second partition index
        models: List of models where models[k] is the model for partition k

    Returns:
        updated_partition_array: Updated partition assignments
    """
    updated_partition_array = partition_array.copy()

    model_i = models[i]
    model_j = models[j]

    if model_i is None or model_j is None:
        return updated_partition_array

    model_i.eval()
    model_j.eval()

    # Process partition j data
    mask_j = partition_array == j
    if mask_j.sum() > 0:
        X_j = X[mask_j]
        y_j = y[mask_j]
        indices_j = np.where(mask_j)[0]

        with torch.no_grad():
            _, logits_i = model_i(X_j)
            preds_i = torch.argmax(logits_i, dim=1)

            _, logits_j = model_j(X_j)
            preds_j = torch.argmax(logits_j, dim=1)

        correct_i = (preds_i == y_j)
        correct_j = (preds_j == y_j)

        # Move to partition i if model_i correct but model_j incorrect
        move_to_i = correct_i & (~correct_j)
        for idx in indices_j[move_to_i.cpu().numpy()]:
            updated_partition_array[idx] = i

    # Process partition i data
    mask_i = partition_array == i
    if mask_i.sum() > 0:
        X_i = X[mask_i]
        y_i = y[mask_i]
        indices_i = np.where(mask_i)[0]

        with torch.no_grad():
            _, logits_i = model_i(X_i)
            preds_i = torch.argmax(logits_i, dim=1)

            _, logits_j = model_j(X_i)
            preds_j = torch.argmax(logits_j, dim=1)

        correct_i = (preds_i == y_i)
        correct_j = (preds_j == y_i)

        # Move to partition j if model_j correct but model_i incorrect
        move_to_j = correct_j & (~correct_i)
        for idx in indices_i[move_to_j.cpu().numpy()]:
            updated_partition_array[idx] = j

    return updated_partition_array


def find_top_removal_potential_image(images, labels, partition_array, models, k, t, batch_size=64, device=None):
    """
    Find the top t data points to remove from partition k that maximize potential decrease.

    Args:
        images: numpy array of images (N, H, W)
        labels: numpy array of labels (N,)
        partition_array: 1D array of integers indicating partition membership
        models: List of models where models[i] is the model for partition i
        k: Working partition index
        t: Number of top points to return
        batch_size: Batch size for DataLoader
        device: Device to use

    Returns:
        top_original_indices: List of t indices in original dataset
        top_updated_models: List of t updated models after each removal
        top_potentials: List of t potential values after each removal
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mask_partition = partition_array == k
    images_partition = images[mask_partition]
    labels_partition = labels[mask_partition]

    dataloader = create_data_loader(images_partition, labels_partition, batch_size, shuffle=False)

    model = models[k]

    original_potential, updated_models, n = compute_potential(model, dataloader)

    # print(f"Dim: {n}, Original potential: {original_potential:.6f}")

    original_indices = np.where(mask_partition)[0]

    potential_decreases = []
    potentials = []

    for i in range(n):
        updated_model = updated_models[i]

        mask = np.ones(n, dtype=bool)
        mask[i] = False
        images_reduced = images_partition[mask]
        labels_reduced = labels_partition[mask]

        dataloader_reduced = create_data_loader(images_reduced, labels_reduced, batch_size, shuffle=False)

        potential_i, _, n_i = compute_potential(updated_model, dataloader_reduced)

        decrease = original_potential - potential_i
        potential_decreases.append(decrease)
        potentials.append(potential_i)

    potential_decreases = np.array(potential_decreases)
    sorted_indices = np.argsort(potential_decreases)[::-1]
    top_t_indices = sorted_indices[:t]

    top_original_indices = [original_indices[i] for i in top_t_indices]

    if t == 1:
        top_updated_models = [updated_models[top_t_indices[0]]]
        top_potentials = [potentials[top_t_indices[0]]]
    else:
        top_updated_models = None
        top_potentials = None

    return top_original_indices, top_updated_models, original_potential


def find_top_removal_influence_image(images, labels, partition_array, models, k, t, batch_size=64, device=None):
    """
    Find top t data points in partition k with highest I(z_i, z_i) values.

    Args:
        images: numpy array of images (N, H, W)
        labels: numpy array of labels (N,)
        partition_array: 1D array of integers indicating partition membership
        models: List of models where models[i] is the model for partition i
        k: Working partition index
        t: Number of top points to return
        batch_size: Batch size for DataLoader
        device: Device to use

    Returns:
        top_indices: List of t indices in original dataset with highest diagonal influence
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from src.Variance import compute_influence_matrix
    mask_partition = partition_array == k
    images_partition = images[mask_partition]
    labels_partition = labels[mask_partition]

    dataloader = create_data_loader(images_partition, labels_partition, batch_size, shuffle=False)

    model = models[k]

    influence_matrix, _, _ = compute_influence_matrix(model, dataloader)
    influence_matrix = influence_matrix.to(device)

    diagonal_influence = -torch.diag(influence_matrix)

    sorted_indices = torch.argsort(diagonal_influence, descending=True)
    top_t_indices = sorted_indices[:t].cpu().numpy()

    original_indices = np.where(mask_partition)[0]

    top_original_indices = original_indices[top_t_indices]

    return top_original_indices.tolist()


def Removal_process_image(images, labels, partition_array, models, max_iterations,
                          test_loader =None, removal_method='top_p', t=10,
                          output_dim=10, num_classes=10, target_loss=0.1, max_epochs=1000, lr=0.001,
                          outlier_indices=None, batch_size=64, device=None):
    """
    Iteratively remove data points from each partition based on different methods.

    Args:
        images: numpy array of images (N, H, W)
        labels: numpy array of labels (N,)
        partition_array: 1D array of integers indicating partition membership
        models: List of models where models[i] is the model for partition i
        max_iterations: Maximum number of iterations per partition
        removal_method: 'seq_p', 'top_p', or 'top_i'
        t: Number of points to remove per iteration (for top_p and top_i methods)
        output_dim: Encoder output dimension
        num_classes: Number of classes
        target_loss: Target training loss
        max_epochs: Maximum training epochs
        lr: Learning rate
        outlier_indices: List/array of outlier indices (if None, no outlier checking)
        batch_size: Batch size for DataLoader
        device: Device to use

    Returns:
        partition_array: Updated partition array with removed points marked as 0
        removal_records: List of tuples (partition_k, iteration, removed_idx, is_outlier)
    """

    partition_array = partition_array.copy()
    removal_records = []

    active_partitions = np.unique(partition_array[partition_array > 0])

    for k in active_partitions:
        for iteration in range(max_iterations):
            if np.sum(partition_array == k) < 10:
                break

            if test_loader is not None:
                acc = evaluate_model(models[k], test_loader)
                # print(f"Model {k}: Accuracy: {acc:.4f}")

            if removal_method == 'seq_p':
                top_indices, top_models, top_potentials = find_top_removal_potential_image(
                    images, labels, partition_array, models, k, t=1, batch_size=batch_size, device=device
                )

                removed_idx = top_indices[0]

                if outlier_indices is not None:
                    is_outlier = int(removed_idx in outlier_indices)
                else:
                    is_outlier = None

                partition_array[removed_idx] = 0
                removal_records.append((k, iteration, removed_idx, is_outlier))

                if outlier_indices is not None:
                    print(f"Partition: {k}, Iteration: {iteration}, Removed Index: {removed_idx}, Potential: {top_potentials[0]:.4f}, Is Outlier: {is_outlier}")

                if top_models is not None:
                    models[k] = top_models[0]

            elif removal_method == 'top_p':
                top_indices, _, original_potential = find_top_removal_potential_image(
                    images, labels, partition_array, models, k, t=t, batch_size=batch_size, device=device
                )

                n = np.sum(partition_array != 0)
                print(f"{iteration},{n},{original_potential:.6f},{acc:.4f}")

                for removed_idx in top_indices:
                    if outlier_indices is not None:
                        is_outlier = int(removed_idx in outlier_indices)
                    else:
                        is_outlier = None

                    partition_array[removed_idx] = 0
                    removal_records.append((k, iteration, removed_idx, is_outlier))

                    if outlier_indices is not None:
                        print(
                            f"Partition: {k}, Iteration: {iteration}, Removed Index: {removed_idx}, Is Outlier: {is_outlier}")

                mask_partition = partition_array == k
                images_partition = images[mask_partition]
                labels_partition = labels[mask_partition]
                dataloader = create_data_loader(images_partition, labels_partition, batch_size, shuffle=False)
                model, _ = train_model_image(dataloader, output_dim, num_classes, target_loss, max_epochs, lr)
                models[k] = model

            elif removal_method == 'top_i':

                mask_partition = partition_array == k
                images_partition = images[mask_partition]
                labels_partition = labels[mask_partition]

                dataloader = create_data_loader(images_partition, labels_partition, batch_size, shuffle=False)

                model = models[k]

                original_potential, _, n = compute_potential(model, dataloader)

                #print(f"Dim: {n}, Original potential: {original_potential:.6f}")
                print(f"{iteration},{n},{original_potential:.6f},{acc:.4f}")

                top_indices = find_top_removal_influence_image(
                    images, labels, partition_array, models, k, t=t, batch_size=batch_size, device=device
                )

                for removed_idx in top_indices:
                    if outlier_indices is not None:
                        is_outlier = int(removed_idx in outlier_indices)
                    else:
                        is_outlier = None

                    partition_array[removed_idx] = 0
                    removal_records.append((k, iteration, removed_idx, is_outlier))

                    if outlier_indices is not None:
                        print(
                            f"Partition: {k}, Iteration: {iteration}, Removed Index: {removed_idx}, Is Outlier: {is_outlier}")

                mask_partition = partition_array == k
                images_partition = images[mask_partition]
                labels_partition = labels[mask_partition]
                dataloader = create_data_loader(images_partition, labels_partition, batch_size, shuffle=False)
                model, _ = train_model_image(dataloader, output_dim, num_classes, target_loss, max_epochs, lr)
                models[k] = model

            elif removal_method == 'rand':
                mask_partition = partition_array == k
                indices_k = np.where(mask_partition)[0]

                n_available = len(indices_k)

                if t > n_available:
                    continue

                selected_indices = np.random.choice(indices_k, t, replace=False)

                images_partition = images[mask_partition]
                labels_partition = labels[mask_partition]

                dataloader = create_data_loader(images_partition, labels_partition, batch_size, shuffle=False)

                model = models[k]

                original_potential, _, n = compute_potential(model, dataloader)

                print(f"{iteration},{n},{original_potential:.6f},{acc:.4f}")
                # print(f"Dim: {n}, Original potential: {original_potential:.6f}")

                for removed_idx in selected_indices:
                    if outlier_indices is not None:
                        is_outlier = int(removed_idx in outlier_indices)
                    else:
                        is_outlier = None

                    partition_array[removed_idx] = 0
                    removal_records.append((k, iteration, removed_idx, is_outlier))

                    if outlier_indices is not None:
                        print(
                            f"Partition: {k}, Iteration: {iteration}, Removed Index: {removed_idx}, Is Outlier: {is_outlier}")

                mask_partition = partition_array == k
                images_partition = images[mask_partition]
                labels_partition = labels[mask_partition]
                dataloader = create_data_loader(images_partition, labels_partition, batch_size, shuffle=False)
                model, _ = train_model_image(dataloader, output_dim, num_classes, target_loss, max_epochs, lr)
                models[k] = model

            else:
                raise ValueError("removal_method must be 'seq_p', 'top_p', or 'top_i'")

    return partition_array, removal_records




