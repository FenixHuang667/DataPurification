
from src.Eval import evaluate_model
from src.LoadImage import create_data_loader, load_images_from_npz, mislabel_data
from src.Variance import compute_potential
import numpy as np
from src.Train import train_model_image


if __name__ == "__main__":

    train_path = "./../data/Image/EMINST_10k_10digits_train.npz"

    test_path = "./../data/Image/EMINST_10k_10digits_test.npz"

    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # digits = [4, 8]
    num_classes = len(digits)
    batch_size = 64
    output_dim = 10
    target_loss = 0.1
    max_epochs = 1000
    lr = 0.001

    # set_determinestic(seed = 42)

    train_images, train_labels = load_images_from_npz(train_path)
    train_loader = create_data_loader(train_images, train_labels, batch_size, shuffle=False)
    print(f"Training data loaded {len(train_images)}")

    test_images, test_labels = load_images_from_npz(test_path)
    test_loader = create_data_loader(test_images, test_labels, batch_size, shuffle=False)
    print(f"Test data loaded  {len(test_images)}")

    for mislabel_fraction in np.arange(0, 0.31, 0.02):
        potentials = []
        accuracies = []

        for i in range(5):
            mislabeled = mislabel_data(train_labels, mislabel_fraction)
            train_loader = create_data_loader(train_images, mislabeled, batch_size, shuffle=False)

            model, final_loss = train_model_image(
                train_loader,
                output_dim,
                num_classes,
                target_loss,
                max_epochs,
                lr)

            potential, _, _ = compute_potential(model, train_loader)
            potentials.append(potential)

            test_accuracy = evaluate_model(model, test_loader)
            accuracies.append(test_accuracy)

        mean_potential = np.mean(potentials)
        std_potential = np.std(potentials)
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        print(
            f"{mislabel_fraction:.2f},{mean_accuracy:.4f},{std_accuracy:.4f},{mean_potential:.4f},{std_potential:.4f}")