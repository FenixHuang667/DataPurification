from itertools import permutations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =====================================================================
# Evaluation: clustering accuracy vs. a known ground-truth partition.
# Cluster labels are arbitrary, so accuracy is taken as the best match
# over all relabelings of the predicted clusters.
# =====================================================================
def partition_accuracy(true_labels, pred_labels, n_components=2):
    """
    Best clustering accuracy over all label permutations.

    Args:
        true_labels: int array (N,) ground-truth partition (e.g. 0 / 1)
        pred_labels: int array (N,) predicted cluster labels
        n_components: number of clusters

    Returns:
        best_acc: float, fraction correct under the best relabeling
        best_perm: tuple, the relabeling pred_label -> true_label used
    """
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    best_acc, best_perm = 0.0, None
    for perm in permutations(range(n_components)):
        mapped = np.array([perm[c] for c in pred_labels])
        acc = float((mapped == true_labels).mean())
        if acc > best_acc:
            best_acc, best_perm = acc, perm
    return best_acc, best_perm


def ground_truth_from_cutpoint(n, cutpoints):
    """
    Build a ground-truth partition array of length n from block boundaries.

    Args:
        n: total number of data points (must equal the number of clustered
           points you will compare against)
        cutpoints: int or list of ints, the sorted block boundaries.
                   One cutpoint c -> points [0, c) get label 0, [c, n) get 1.
                   Multiple cutpoints -> each successive segment gets 0, 1, 2, ...

    Returns:
        g: int array (n,) with block labels 0, 1, ...
    """
    if np.isscalar(cutpoints):
        cutpoints = [cutpoints]
    bounds = [0] + list(cutpoints) + [n]
    g = np.zeros(n, dtype=int)
    for b in range(len(bounds) - 1):
        g[bounds[b]:bounds[b + 1]] = b
    return g


def adjusted_rand_index(true_labels, pred_labels):
    """
    Adjusted Rand Index between a ground-truth partition and a clustering.

    Chance-corrected and invariant to label permutation, so no cluster-to-block
    matching is required. Range: ~0 for random agreement, 1 for identical
    partitions (can be slightly negative for worse-than-chance).

    Args:
        true_labels: int array (N,) ground-truth block labels
        pred_labels: int array (N,) predicted cluster labels

    Returns:
        ari: float
    """
    true = np.asarray(true_labels)
    pred = np.asarray(pred_labels)
    if len(true) != len(pred):
        raise ValueError(
            f"length mismatch: true has {len(true)} labels but pred has "
            f"{len(pred)}. Build the ground-truth vector to match the number "
            f"of clustered points (e.g. ground_truth_from_cutpoint(len(pred), c))."
        )

    # contingency table n_ij = # points with true==i and pred==j
    true_ids = {v: i for i, v in enumerate(np.unique(true))}
    pred_ids = {v: j for j, v in enumerate(np.unique(pred))}
    table = np.zeros((len(true_ids), len(pred_ids)), dtype=np.int64)
    for t, p in zip(true, pred):
        table[true_ids[t], pred_ids[p]] += 1

    def comb2(x):
        return x * (x - 1) / 2.0

    sum_comb_cells = comb2(table).sum()
    sum_comb_rows = comb2(table.sum(axis=1)).sum()
    sum_comb_cols = comb2(table.sum(axis=0)).sum()
    total_comb = comb2(table.sum())

    expected = sum_comb_rows * sum_comb_cols / total_comb
    max_index = 0.5 * (sum_comb_rows + sum_comb_cols)
    denom = max_index - expected
    if denom == 0:          # both partitions trivial (single cluster)
        return 1.0
    return float((sum_comb_cells - expected) / denom)


def evaluate_partition(true_labels, pred_labels, n_components=2):
    """
    Per-block evaluation of a clustering against a ground-truth partition.

    Cluster identities are arbitrary, so predicted clusters are first relabeled
    to true blocks by the permutation that maximizes total agreement. Then, for
    each true block b:

        tp_b        = # points with true == b AND matched-pred == b
        support_b   = # points with true == b           (size of true block b)
        pred_size_b = # points with matched-pred == b    (size of predicted block b)

        accuracy_b  = recall_b = tp_b / support_b
                      (correctly identified in block b / total data in block b)
        precision_b = tp_b / pred_size_b
        f1_b        = 2 * precision_b * recall_b / (precision_b + recall_b)

    Args:
        true_labels: int array (N,) ground-truth partition (e.g. 0 / 1)
        pred_labels: int array (N,) predicted cluster labels
        n_components: number of blocks

    Returns:
        results: dict block -> {accuracy, recall, precision, f1,
                                support, predicted_size, tp}
        overall_accuracy: float, total correct / N under the matching
        mapping: dict predicted_cluster -> true_block (the chosen relabeling)
    """
    true = np.asarray(true_labels)
    pred = np.asarray(pred_labels)
    N = len(true)

    if len(pred) != N:
        raise ValueError(
            f"length mismatch: true has {N} labels but pred has {len(pred)}. "
            f"The ground-truth vector must have one entry per clustered point. "
            f"Build it for the actual data size, e.g. "
            f"ground_truth_from_cutpoint(len(pred), cutpoint)."
        )

    # best relabeling of predicted clusters -> true blocks
    best_perm, best_correct = None, -1
    for perm in permutations(range(n_components)):
        mapped = np.array([perm[c] for c in pred])
        correct = int((mapped == true).sum())
        if correct > best_correct:
            best_correct, best_perm = correct, perm
    mapped = np.array([best_perm[c] for c in pred])

    results = {}
    for b in range(n_components):
        tp = int(((true == b) & (mapped == b)).sum())
        support = int((true == b).sum())          # total data in true block b
        pred_size = int((mapped == b).sum())      # data assigned to block b
        recall = tp / support if support else 0.0          # = per-block accuracy
        precision = tp / pred_size if pred_size else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        results[b] = {
            "accuracy": recall,        # correctly identified / total in block b
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "support": support,
            "predicted_size": pred_size,
            "tp": tp,
        }

    overall_accuracy = best_correct / N
    mapping = {c: best_perm[c] for c in range(n_components)}
    return results, overall_accuracy, mapping


def print_evaluation(results, overall_accuracy, mapping, title="evaluation"):
    """
    Pretty-print a per-block evaluation, including the size of each predicted
    cluster (n_clustered) alongside the true block size (support).
    """
    print(f"\n=== {title} ===   overall_accuracy={overall_accuracy:.4f}   "
          f"pred_cluster->true_block map={mapping}")
    print(f"{'block':>5} {'support':>8} {'n_clustered':>12} "
          f"{'accuracy':>9} {'precision':>10} {'f1':>7}")
    for b, m in results.items():
        print(f"{b:>5} {m['support']:>8} {m['predicted_size']:>12} "
              f"{m['accuracy']:>9.4f} {m['precision']:>10.4f} {m['f1']:>7.4f}")


# =====================================================================
# Loader: combine the 10 question features + the label into 11 columns,
# encoded as integer category codes (NOT one-hot).
# =====================================================================
def load_voting_combined(csv_file, feature_cols, label_col):
    """
    Load voting CSV and combine features + label into one (N, 11) integer
    matrix of category codes, one column per variable.

    Returns:
        data: int array (N, D) of category codes (D = len(feature_cols)+1)
        n_categories: list of int, number of categories per column
    """
    df = pd.read_csv(csv_file)
    cols = list(feature_cols) + [label_col]

    data = np.zeros((len(df), len(cols)), dtype=int)
    n_categories = []
    for j, col in enumerate(cols):
        cats = sorted(df[col].unique())
        mapping = {v: i for i, v in enumerate(cats)}
        data[:, j] = df[col].map(mapping).values
        n_categories.append(len(cats))

    return data, n_categories


# =====================================================================
# (1) Unsupervised: EM-based mixture model (categorical / latent class).
#
#     p(x_i) = sum_k pi_k * prod_d theta[k, d, x_i,d]
#     Each of the 11 categorical variables is conditionally independent
#     given the latent component. Standard EM in log space.
# =====================================================================
def em_mixture_categorical(data, n_components=2, n_categories=None,
                           max_iter=200, tol=1e-6, smoothing=1e-3, seed=0):
    """
    EM for a mixture of categorical (latent-class) distributions.

    Args:
        data: int array (N, D) of category codes (use load_voting_combined)
        n_components: number of mixture components (2 for f1, f2)
        n_categories: list of category counts per column (inferred if None)
        max_iter, tol: EM stopping criteria
        smoothing: Laplace smoothing on category probabilities
        seed: RNG seed for responsibility initialization

    Returns:
        assignments: int array (N,) hard component label in [0, K-1]
        responsibilities: float array (N, K)
        params: dict with pi, theta (list of (K, C_d) arrays), log_lik, n_iter
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=int)
    N, D = data.shape
    if n_categories is None:
        n_categories = [int(data[:, d].max()) + 1 for d in range(D)]

    # init: random responsibilities, then start with M-step
    r = rng.dirichlet(np.ones(n_components), size=N)  # (N, K)

    pi = None
    theta = None
    log_lik = -np.inf
    log_lik_old = -np.inf
    it = 0
    for it in range(max_iter):
        # ---- M-step ----
        Nk = r.sum(axis=0) + 1e-12          # (K,)
        pi = Nk / N
        theta = []
        for d in range(D):
            Cd = n_categories[d]
            counts = np.zeros((n_components, Cd))
            for c in range(Cd):
                indicator = (data[:, d] == c).astype(float)   # (N,)
                counts[:, c] = r.T @ indicator                # (K,)
            counts += smoothing
            theta.append(counts / counts.sum(axis=1, keepdims=True))

        # ---- E-step (log space) ----
        log_r = np.tile(np.log(pi)[None, :], (N, 1))          # (N, K)
        for d in range(D):
            log_theta_d = np.log(theta[d])                    # (K, C_d)
            log_r += log_theta_d[:, data[:, d]].T             # (N, K)

        m = log_r.max(axis=1, keepdims=True)
        log_norm = m + np.log(np.exp(log_r - m).sum(axis=1, keepdims=True))
        log_lik = float(log_norm.sum())
        r = np.exp(log_r - log_norm)

        if abs(log_lik - log_lik_old) < tol * (abs(log_lik_old) + 1e-12):
            break
        log_lik_old = log_lik

    assignments = r.argmax(axis=1)
    params = {"pi": pi, "theta": theta, "log_lik": log_lik, "n_iter": it + 1}
    return assignments, r, params


# =====================================================================
# (2) Supervised: Mixture of Experts (MoE).
#
#     gate g(x) = softmax(W_g x)            -> soft assignment over experts
#     expert_k:  p_k(y|x) = softmax(W_k x)  -> per-expert classifier
#     p(y|x) = sum_k g_k(x) * p_k(y|x)
#
#     Trained by minimizing the marginal NLL. Use one-hot X (e.g. from
#     LoadSyn.load_voting_csv) as input.
# =====================================================================
class MoE(nn.Module):
    def __init__(self, input_size, num_classes, num_experts=2):
        super().__init__()
        self.gate = nn.Linear(input_size, num_experts)
        self.experts = nn.ModuleList(
            [nn.Linear(input_size, num_classes) for _ in range(num_experts)]
        )
        self.num_experts = num_experts

    def forward(self, x):
        log_gate = torch.log_softmax(self.gate(x), dim=1)                 # (N, K)
        expert_log_probs = torch.stack(
            [torch.log_softmax(e(x), dim=1) for e in self.experts], dim=1
        )                                                                 # (N, K, C)
        return log_gate, expert_log_probs


def moe_nll(log_gate, expert_log_probs, y):
    """Marginal negative log-likelihood: -log sum_k g_k p_k(y|x)."""
    y_idx = y.view(-1, 1, 1).expand(-1, expert_log_probs.size(1), 1)
    expert_ll = expert_log_probs.gather(2, y_idx).squeeze(2)             # (N, K)
    joint = log_gate + expert_ll                                         # (N, K)
    return -torch.logsumexp(joint, dim=1).mean()


def train_moe(X, y, input_size, num_classes, num_experts=2,
              lr=0.01, max_epochs=500, batch_size=64, verbose=False, seed=0):
    """
    Train a MoE on (X, y). X should be the one-hot feature tensor and y the
    integer label tensor (same format as LoadSyn.load_voting_csv output).

    Returns:
        model: trained MoE
        final_loss: last-epoch average NLL
    """
    torch.manual_seed(seed)
    model = MoE(input_size, num_classes, num_experts)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    final_loss = float("nan")
    for epoch in range(max_epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            log_gate, expert_log_probs = model(xb)
            loss = moe_nll(log_gate, expert_log_probs, yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        final_loss = total / len(loader)
        if verbose and epoch % 50 == 0:
            print(f"epoch {epoch} nll {final_loss:.4f}")

    return model, final_loss


def moe_assignments(model, X, y=None):
    """
    Get expert assignments from a trained MoE.

    Returns:
        gate_assign: argmax of the gate g(x)            (N,)
        post_assign: argmax of posterior g_k p_k(y|x)   (N,) or None if y is None
        responsibilities: gate probs, or posterior probs if y is given  (N, K)
    """
    model.eval()
    with torch.no_grad():
        log_gate, expert_log_probs = model(X)
        gate_assign = log_gate.argmax(1).numpy()
        if y is not None:
            y_idx = y.view(-1, 1, 1).expand(-1, expert_log_probs.size(1), 1)
            expert_ll = expert_log_probs.gather(2, y_idx).squeeze(2)
            post = torch.softmax(log_gate + expert_ll, dim=1)
            return gate_assign, post.argmax(1).numpy(), post.numpy()
        return gate_assign, None, torch.softmax(log_gate, dim=1).numpy()