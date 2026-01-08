import torch
from src.Eval import evaluate_model
from src.LoadSyn import load_voting_csv,create_dataloader, extract_partition_dataloader
import numpy as np
from src.Train import train_logistic_regression
from src.Purify import Removal_process_vote


def generate_random_partition(n, k):
    """
    Generate a random partition array.

    Args:
        n: Number of data points
        k: Number of partitions

    Returns:
        partition: Array of length n with random integers between 1 and k
    """
    partition = np.random.randint(1, k + 1, size=n)

    return partition


def route_and_predict(X, y, router_model, models):
    """
    Route data and predict with per-partition accuracy metrics.

    Returns:
        predictions: Overall predictions
        partition_assignments: Partition assignments
        partition_accuracies: Dictionary with accuracy for each partition
    """
    router_model.eval()

    with torch.no_grad():
        _, router_logits = router_model(X)
        partition_assignments = torch.argmax(router_logits, dim=1)

    predictions = torch.zeros(len(X), dtype=torch.long)
    partition_accuracies = {}

    for partition_idx in range(len(models)):
        if models[partition_idx] is None:
            continue

        mask = partition_assignments == partition_idx
        if mask.sum() == 0:
            continue

        X_partition = X[mask]
        y_partition = y[mask]

        models[partition_idx].eval()
        with torch.no_grad():
            _, logits = models[partition_idx](X_partition)
            preds = torch.argmax(logits, dim=1)

        predictions[mask] = preds

        accuracy = (preds == y_partition).sum().item() / len(y_partition)
        partition_accuracies[partition_idx] = accuracy

    return predictions, partition_assignments, partition_accuracies

if __name__ == "__main__":
    #data_path = "./../data/Synthetic/synthetic_train_pure.csv"  #Pure distribution 1
    data_path = "./../data/Synthetic/synthetic_train_mix_2spec.csv" # 7:3 mix distribution 1 and 2
    #data_path = "./../data/Synthetic/synthetic_train_mix_3spec.csv" #7:1.5:1.5 mix distribution 1, 2, and 3

    test_path = "./../data/Synthetic/synthetic_test_dis1.csv"


    feature_columns = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9']
    label_columns = 'Q10'
    batch_size = 64


    X_full, y_full = load_voting_csv(
        csv_file=data_path,
        feature_cols=feature_columns,
        label_col=label_columns,
    )
    print(f"Training data loaded {len(X_full)}")

    # Infer input_size and num_classes
    input_size = X_full.shape[1]
    num_classes = 4

    data_loader = create_dataloader(X_full, y_full, batch_size)

    X_test, y_test = load_voting_csv(
        csv_file=test_path,
        feature_cols=feature_columns,
        label_col=label_columns,
    )
    print(f"Test data loaded {len(X_test)}")

    test_loader = create_dataloader(X_test, y_test, batch_size)

    full_model, _ = train_logistic_regression(data_loader, input_size, num_classes)
    accuracy_fm = evaluate_model(full_model, test_loader)

    print(f"Full model acc: {accuracy_fm}")


    parition = generate_random_partition(len(X_full), 1)
    
    data_loader_sub1 = extract_partition_dataloader(X_full, y_full, parition, 1, batch_size)

    model1, loss = train_logistic_regression(data_loader_sub1, input_size, num_classes)
    accuracy_m1 = evaluate_model(model1, test_loader)

    print(f"S1:{len(data_loader_sub1.dataset)}  Model 1 acc: {accuracy_m1:.4f}  loss: {loss:.4f}")


    models = [None, model1]

    parition, removal_records = Removal_process_vote(
        X_full, y_full, parition, models, max_iterations=90,
        test_loader= test_loader, removal_method='top_p', t=10,
        input_size=input_size, num_classes=num_classes,
        batch_size=batch_size
    )