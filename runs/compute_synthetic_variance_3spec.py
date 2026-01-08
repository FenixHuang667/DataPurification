from src.LoadSyn import load_voting_csv, create_dataloader
from src.Eval import split_train_test_vote, evaluate_model
from src.Variance import compute_potential
from src.Train import train_logistic_regression
import numpy as np
import torch


def subsample_from_three_distributions(X, y, k1, k2, n, c0, c1, c2, seed=None):
    """
    Subsample data from three distributions.

    Args:
        X: Feature tensor of shape (N, num_features)
        y: Label tensor of shape (N,)
        k1: End index of first distribution (0-based, inclusive)
        k2: End index of second distribution (0-based, inclusive)
        n: End index of third distribution (0-based, inclusive)
        c0: Number of samples from [0, k1]
        c1: Number of samples from [k1+1, k2]
        c2: Number of samples from [k2+1, n]
        seed: Random seed for reproducibility

    Returns:
        X_sub: Subsampled features
        y_sub: Subsampled labels
        selected_indices: Indices of selected samples
    """
    if seed is not None:
        torch.manual_seed(seed)

    dist1_indices = torch.arange(0, k1 + 1)
    dist2_indices = torch.arange(k1 + 1, k2 + 1)
    dist3_indices = torch.arange(k2 + 1, n + 1)

    if c0 > len(dist1_indices):
        raise ValueError(f"Not enough samples in first distribution to pick {c0}.")
    if c1 > len(dist2_indices):
        raise ValueError(f"Not enough samples in second distribution to pick {c1}.")
    if c2 > len(dist3_indices):
        raise ValueError(f"Not enough samples in third distribution to pick {c2}.")

    indices_0 = dist1_indices[torch.randperm(len(dist1_indices))[:c0]]
    indices_1 = dist2_indices[torch.randperm(len(dist2_indices))[:c1]]
    indices_2 = dist3_indices[torch.randperm(len(dist3_indices))[:c2]]

    selected_indices = torch.cat([indices_0, indices_1, indices_2])

    X_sub = X[selected_indices]
    y_sub = y[selected_indices]

    return X_sub, y_sub, selected_indices


if __name__ == "__main__":
    test_path = "./../data/Synthetic/synthetic_mix123.csv"
    feature_columns = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9']
    label_columns = 'Q10'
    batch_size = 64
    max_epochs = 2000
    total_data = 600

    X_full, y_full = load_voting_csv(
        csv_file=test_path,
        feature_cols=feature_columns,
        label_col=label_columns,
    )
    print("Synthetic data loaded")

    input_size = X_full.shape[1]
    num_classes = 4

    k1 = 999
    k2 = 1999
    n = 2999

    for a in range(0, total_data + 1, 50):
        for b in range(0, total_data - a + 1, 50):
            c0 = total_data - a - b
            c1 = a
            c2 = b

            repeats = 5
            accuracies = []
            potentials = []

            for i in range(repeats):
                X_sub, y_sub, subset_indices = subsample_from_three_distributions(
                    X_full, y_full, k1, k2, n, c0, c1, c2
                )

                train_loader, test_loader = split_train_test_vote(
                    X_sub, y_sub, 0.8, batch_size, shuffle=True
                )

                train_model, train_final_loss = train_logistic_regression(
                    train_loader, input_size, num_classes, max_epochs=max_epochs
                )
                test_accuracy = evaluate_model(train_model, test_loader)

                accuracies.append(test_accuracy)

                data_loader = create_dataloader(X_sub, y_sub, batch_size)
                model, final_loss = train_logistic_regression(
                    data_loader, input_size, num_classes, max_epochs=max_epochs
                )
                potential, _, _ = compute_potential(model, data_loader)

                potentials.append(potential)

            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies, ddof=1)

            avg_p = np.mean(potentials)
            std_p = np.std(potentials, ddof=1)

            print(f"{a},{b},{avg_acc:.4f},{std_acc:.4f},{avg_p:.4f},{std_p:.4f}")