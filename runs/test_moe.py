import numpy as np
from src.LoadSyn import load_voting_csv
from src.Cluster import (load_voting_combined, em_mixture_categorical,
                         adjusted_rand_index, ground_truth_from_cutpoint)

feat=['Q0','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9']; lab='Q10'
csv="./../data/Synthetic/synthetic_train_mix_2spec.csv"

"""
# ground truth: first 700 -> distribution 0, the rest -> distribution 1
combined, ncat = load_voting_combined(csv, feat, lab)
N = len(combined)
true = np.ones(N, dtype=int); true[:700] = 0

true = np.ones(1000, dtype=int); true[:700] = 0   # 0-699 -> dist 1, rest -> dist 2

# EM
combined, ncat = load_voting_combined(csv, feat, lab)
assign, _, _ = em_mixture_categorical(combined, 2, ncat)
res, overall, mapping = evaluate_partition(true, assign, 2)
for b, m in res.items():
    print(f"block {b}: acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  (n={m['support']})")
"""

# EM-based clustering on the 11 combined categorical features
combined, ncat = load_voting_combined(csv, feat, lab)
assign, _, _ = em_mixture_categorical(combined, n_components=2, n_categories=ncat)

# ground truth: 0-699 -> block 0, 700-999 -> block 1
true = ground_truth_from_cutpoint(len(assign), 700)

ari = adjusted_rand_index(true, assign)
print(f"EM cluster sizes: {np.bincount(assign)}")
print(f"Adjusted Rand Index: {ari:.4f}")


"""
# (2) MoE
X, y = load_voting_csv(csv, feat, lab)
model, _ = train_moe(X, y, X.shape[1], num_classes=4, num_experts=2)
gate_a, post_a, _ = moe_assignments(model, X, y)
print("MoE gate :", partition_accuracy(true, gate_a)[0])
print("MoE post :", partition_accuracy(true, post_a)[0])
"""


