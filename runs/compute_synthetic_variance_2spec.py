
from src.LoadSyn import load_voting_csv, subsample_from_partitions, create_dataloader
from src.Eval import split_train_test_vote, evaluate_model
from src.Variance import compute_potential
from src.Train import train_logistic_regression
import numpy as np

if __name__ == "__main__":
    test_path = "./../data/Synthetic/synthetic_mix12.csv"
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

    # Infer input_size and num_classes
    input_size = X_full.shape[1]
    num_classes = 4

    ############Train-test split
    for a in range(0,total_data+1, 20):
        repeats = 5

        accuracies = []
        potentials = []
        for i in range(repeats):
            # select data set
            X_sub, y_sub, subset_indices = subsample_from_partitions(X_full, y_full, 999, total_data-a, a)

            # run a train-test split for test acc
            train_loader ,test_loader = split_train_test_vote(X_sub, y_sub, 0.8, batch_size, shuffle=True)

            train_model, train_final_loss = train_logistic_regression(train_loader, input_size, num_classes, max_epochs=max_epochs)
            test_accuracy = evaluate_model(train_model, test_loader)

            accuracies.append(test_accuracy)

            #compute the potential for entire set
            data_loader = create_dataloader(X_sub, y_sub, batch_size)
            model, final_loss = train_logistic_regression(data_loader, input_size, num_classes,  max_epochs=max_epochs)
            potential, _, _ = compute_potential(model, data_loader)

            potentials.append(potential)

            # print(f"Run {i + 1}: acc = {test_accuracy:.4f}  p = {potential:.4f}")

        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1)  # sample standard deviation

        avg_p = np.mean(potentials)
        std_p = np.std(potentials, ddof=1)  # sample standard deviation

        #print(f"\nAverage accuracy over {repeats} runs = {avg_acc:.4f}")
        #print(f"Standard deviation = {std_acc:.4f}")

        #print(f"\nAverage potential over {repeats} runs = {avg_p:.4f}")
        #print(f"Standard deviation = {std_p:.4f}")

        print(f"{a},{avg_acc:.4f},{std_acc:.4f},{avg_p:.4f},{std_p:.4f}")