
from src.Eval import evaluate_model
from src.LoadImage import create_data_loader, load_images_from_npz, extract_partition_dataloader_image
import numpy as np
from src.Train import train_model_image
from src.Purify import Removal_process_image



if __name__ == "__main__":

    train_path = "./../data/Image/Image_train_30error.npz"
    test_path = "./../data/Image/Image_test.npz"

    digits = [4, 8]
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
    print(f"Test data loaded {len(test_images)}")

    model, final_loss = train_model_image(
        train_loader,
        output_dim,
        num_classes,
        target_loss,
        max_epochs,
        lr)

    test_accuracy = evaluate_model(model, test_loader)

    print(f"Test acc: {test_accuracy:.6f}")

    partition_array = np.ones(len(train_images), dtype=int)

    models = [None, model]

    partition_array, removal_records = Removal_process_image(
        train_images, train_labels, partition_array, models, test_loader=test_loader, max_iterations=50,
        removal_method='top_p', t=10,
        output_dim=output_dim, num_classes=num_classes, target_loss=target_loss, max_epochs=max_epochs, lr=lr,
        batch_size=batch_size
    )

    data_loader_p1 = extract_partition_dataloader_image(train_images, train_labels, partition_array, 1, batch_size)

    model_p1, _ = train_model_image(
        data_loader_p1,
        output_dim,
        num_classes,
        target_loss,
        max_epochs,
        lr)

    test_accuracy = evaluate_model(model_p1, test_loader)

    print(f"P1 Test acc: {test_accuracy:.6f}")