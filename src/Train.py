import torch
import torch.nn as nn
import torch.optim as optim


def set_determinestic(seed: int=42):
    ## make the model stable
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


### Training and evaluate model
#Manually switch model between LeNet and Logistic regression (LG)
def train_model_image(train_loader, output_dim=10, num_classes=10, target_loss=0.1,
                max_epochs=100, lr=0.001):
    """
    Train encoder and classifier model until target loss or max epochs.

    Args:
        train_loader: DataLoader for training data
        output_dim: Dimension of encoder output
        num_classes: Number of output classes
        target_loss: Stop when loss reaches this value
        max_epochs: Maximum training epochs
        lr: Learning rate

    Returns:
        model: Trained model (Classifier with encoder)
        final_loss: Final training loss
    """

    # set_determinestic(seed = 42)

    from src.LoadImage import LeNetEncoder, Classifier, LogisticRegressionImage

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Use LetNet
    #encoder = LeNetEncoder(output_dim=output_dim).to(device)
    #model = Classifier(encoder, num_classes=num_classes).to(device)

    #Use logistic regression
    model = LogisticRegressionImage(28*28, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    epoch_loss = 0.0

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            features, outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() #average loss of a batch

        epoch_loss = running_loss / len(train_loader) # average over all batches

        #if epoch % 20 == 0:
        #    print(f"Epoch {epoch} loss: {epoch_loss}")

        if epoch_loss <= target_loss:
            #print(f'Target loss reached at epoch {epoch + 1}')
            break

    final_loss = epoch_loss
    return model, final_loss


#Main, training for voting data
def train_logistic_regression(dataloader, input_size, num_classes,
                              target_loss=0.1, lr=0.001, max_epochs=1000, verbose=False):
    """
    Train a logistic regression model until training loss reaches target.

    Args:
        dataloader: DataLoader for training data
        input_size: Number of input features
        num_classes: Number of output classes
        target_loss: Target training loss
        lr: Learning rate
        max_epochs: Maximum training epochs
        verbose: Print progress

    Returns:
        model: Trained LogisticRegression model
        final_loss: Final training loss
    """
    from src.LoadSyn import LogisticRegression, MLPVote

    # set_determinestic(seed = 42)

    model = LogisticRegression(input_size, num_classes)
    # model = MLPVote(input_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(max_epochs):
        epoch_loss = 0.0

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            features, outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(dataloader)

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

        if epoch_loss <= target_loss:
            if verbose:
                print(f"Target loss {target_loss} reached at epoch {epoch}")
            break

    final_loss = epoch_loss
    return model, final_loss

