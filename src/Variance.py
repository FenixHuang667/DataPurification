import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import copy

# ============================================================
# Utility 1: Recover full dataset from a DataLoader
# ============================================================
def collect_dataset_from_dataloader(dataloader, device=None):
    all_X, all_y = [], []
    for X_batch, y_batch in dataloader:
        all_X.append(X_batch)
        all_y.append(y_batch)
    X = torch.cat(all_X)
    y = torch.cat(all_y)
    if device is not None:
        X, y = X.to(device), y.to(device)
    return X, y

# ============================================================
# Utility 2: Compute per-sample gradients
# ============================================================
def compute_sample_gradients(model, X, y, criterion=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    grads = []
    for i in range(len(X)):
        model.zero_grad()
        #output = model(X[i].unsqueeze(0))
        _, output = model(X[i].unsqueeze(0))
        loss = criterion(output, y[i].unsqueeze(0))
        grad = torch.autograd.grad(loss, model.parameters(), create_graph=False)
        grad_flat = torch.cat([g.reshape(-1) for g in grad])
        grads.append(grad_flat.detach())

    return torch.stack(grads)  # [n_samples, n_params]

# ============================================================
# Utility 3: Compute full Hessian explicitly
# ============================================================
#Input X,y as full data to compute the Hessian
#Input X[i].unsqueeze(0), y[i].unsqueeze(0) to compute \nabla^2 L(z_i, \hat\theta)
def compute_full_hessian(model, X, y, criterion=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    model.zero_grad()
    # outputs = model(X)
    _, outputs = model(X)
    total_loss = criterion(outputs, y)
    first_grads = torch.autograd.grad(total_loss, model.parameters(), create_graph=True)
    grad_vector = torch.cat([g.reshape(-1) for g in first_grads])
    n_params = grad_vector.numel()

    H_rows = []
    for i in range(n_params):
        #if i % 1000 == 0:
        #    print(f"H inverse row {i}")
        second_grad = torch.autograd.grad(grad_vector[i], model.parameters(), retain_graph=True)
        H_row = torch.cat([g.reshape(-1) for g in second_grad])
        # H_rows.append(H_row.detach())
        H_rows.append(H_row)
    return torch.stack(H_rows)  # [n_params, n_params]



# ============================================================
# Utility 4: Invert Hessian with regularization
# ============================================================
def invert_hessian(H, damping=0.01):
    n_params = H.shape[0]
    H_reg = H + damping * torch.eye(n_params, device=H.device)

    # H_inv = torch.inverse(H_reg)
    H_inv = torch.linalg.inv(H_reg)

    return H_inv



# ============================================================
# Utility 5: Compute influence matrix
# ============================================================


def compute_influence_matrix_internal(grads, H_inv):
    # grads: [n_samples, n_params]
    # H_inv: [n_params, n_params]

    Hg = H_inv @ grads.T  # [n_params, n_samples]
    influence_matrix = -grads @ Hg  # [n_samples, n_samples]

    return influence_matrix



# ============================================================
# High-level orchestration
# ============================================================
def compute_influence_matrix(model, dataloader, mask=None, device=None, damping=0.01):
    """
    Compute influence matrix considering only unmasked data points.

    Args:
        model: Trained model
        dataloader: DataLoader for the entire dataset
        mask: Boolean array indicating which points are active (True = active, False = masked/removed)
              If None, all points are considered active
        device: Device to use
        damping: Damping parameter for Hessian inversion

    Returns:
        influence_matrix: Influence matrix for unmasked points only
        updated_models: List of models with each unmasked point removed
        grads: Gradients for unmasked points only
    """
    if device is None:
        device = next(model.parameters()).device

    X, y = collect_dataset_from_dataloader(dataloader, device=device)

    if mask is None:
        mask = np.ones(len(X), dtype=bool)

    X_masked = X[mask]
    y_masked = y[mask]

    grads = compute_sample_gradients(model, X_masked, y_masked).to(device)

    H = compute_full_hessian(model, X_masked, y_masked)

    H_inv = invert_hessian(H, damping=damping)

    updated_models = update_model(model, grads, H_inv, device)

    influence_matrix = compute_influence_matrix_internal(grads, H_inv)

    return influence_matrix.cpu(), updated_models, grads


def compute_potential(model, dataloader):

    influence_matrix, updated_models, grads = compute_influence_matrix(model, dataloader)

    n = influence_matrix.shape[0]

    # Vectorized: compute sum of all squared elements
    total = torch.sum(influence_matrix ** 2)

    """
    # compute the second moment
    # Subtract diagonal elements
    diagonal_sum = torch.sum(torch.diag(influence_matrix) ** 2)
    total_squared_influence = total - diagonal_sum
    potential = total_squared_influence / (n * (n - 1))
    # print("Potential done")
    """

    # compute the variance of off-diagonals
    mask = ~torch.eye(n, dtype=torch.bool, device=influence_matrix.device)
    values = influence_matrix[mask]
    # population variance
    var_offdiag = torch.var(values, unbiased=False)

    potential = var_offdiag.item()

    return potential, updated_models, n



def update_model(model, grads, H_inv, device=None):
    """
    Update model parameters by removing each data point using influence functions.

    Based on Koh & Liang (2017):
    θ_{-z} = θ - H^(-1) * grad_z

    Args:
        model: Trained model
        grads: Pre-computed gradients for all data points [n_samples, n_params]
        H_inv: Pre-computed inverse Hessian matrix [n_params, n_params]
        device: Device to use

    Returns:
        updated_models: List of models, each with one data point removed
    """
    if device is None:
        device = next(model.parameters()).device

    grads = grads.to(device)
    H_inv = H_inv.to(device)

    n_samples = grads.shape[0]
    updated_models = []
    """
    # Get model structure info and original state once before the loop for logistic regression
    input_size = model.linear.in_features
    num_classes = model.linear.out_features

    # Get model structure info and original state once before the loop for MLP
    input_size = model.fc1.in_features
    num_classes = model.fc4.out_features


    original_state_dict = model.state_dict()  # Capture the original parameters
    """


    # 2. Iteration and Update
    for i in range(n_samples):
        grad_z = grads[i]  # p-dimensional vector

        # Compute the influence term: H^{-1} * grad L(z_i, theta)
        H_inv_grad_z = H_inv @ grad_z

        """
        # Create a fresh copy of the model with original parameters for logistic regression
        updated_model = LogisticRegression(input_size, num_classes)
        updated_model.load_state_dict(original_state_dict)
        updated_model.to(device)

        # Create a fresh copy of the model with original parameters for MLP
        updated_model = MLPVote(input_size, num_classes)
        updated_model.load_state_dict(original_state_dict)
        updated_model.to(device)
        """

        updated_model = copy.deepcopy(model)
        updated_model.to(device)

        # Apply the update (theta_{-z} = theta - H^{-1} * grad L(z, theta))
        offset = 0
        for name, param in updated_model.named_parameters():
            if param.requires_grad:
                param_size = param.numel()
                # Slice the influence term vector and reshape it to match the parameter's shape
                param_update = (-1.0) * H_inv_grad_z[offset:offset + param_size].view(param.shape) / n_samples

                # Update the parameter: param.data = param.data - param_update
                param.data.sub_(param_update)  # Use in-place subtraction for efficiency

                offset += param_size

        updated_models.append(updated_model)

    return updated_models
