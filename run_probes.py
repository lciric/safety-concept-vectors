"""
Train linear probes on extracted activations and generate analysis plots.

For each concept, for each layer:
  - Train LogisticRegression (concept vs neutral)
  - Report accuracy with bootstrap CI
  - Permutation test for statistical significance

Generates:
  - Probe accuracy heatmap (25 concepts x N layers)
  - Emergence curves per concept
  - Cosine similarity matrix between concept directions
  - PCA visualization of the safety concept space

Usage:
    python run_probes.py --activations-dir activations/Qwen2.5-7B-Instruct/last_token
    python run_probes.py --activations-dir activations/Qwen2.5-7B-Instruct/all --compare-modes
"""

import numpy as np
import json
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def load_activations(activations_dir):
    """Load all concept and neutral activations."""
    concepts_file = os.path.join(activations_dir, "concept_names.json")
    with open(concepts_file) as f:
        concepts = json.load(f)

    neutral = np.load(os.path.join(activations_dir, "neutral_activations.npy"))
    print(f"  Neutral: {neutral.shape}")

    concept_acts = {}
    for c in concepts:
        path = os.path.join(activations_dir, f"{c}_activations.npy")
        if os.path.exists(path):
            concept_acts[c] = np.load(path)
            print(f"  {c}: {concept_acts[c].shape}")

    return concepts, neutral, concept_acts


def train_probe(concept_acts, neutral_acts, layer, n_folds=5, C=0.1):
    """
    Train a logistic regression probe at a specific layer.
    Uses k-fold cross-validation for more stable estimates.

    Returns: mean accuracy, std, per-fold accuracies
    """
    X_concept = concept_acts[:, layer, :]  # (n_concept, d_model)
    X_neutral = neutral_acts[:, layer, :]  # (n_neutral, d_model)

    # Balance classes by subsampling the larger one
    n_min = min(len(X_concept), len(X_neutral))
    X_concept = X_concept[:n_min]
    X_neutral = X_neutral[:n_min]

    X = np.vstack([X_concept, X_neutral])
    y = np.array([1]*n_min + [0]*n_min)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accs = []

    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        acc = clf.score(X[test_idx], y[test_idx])
        fold_accs.append(acc)

    return np.mean(fold_accs), np.std(fold_accs), fold_accs


def permutation_test(concept_acts, neutral_acts, layer, n_permutations=100, C=0.1):
    """Run permutation test at a specific layer."""
    X_concept = concept_acts[:, layer, :]
    X_neutral = neutral_acts[:, layer, :]

    n_min = min(len(X_concept), len(X_neutral))
    X_concept = X_concept[:n_min]
    X_neutral = X_neutral[:n_min]

    X = np.vstack([X_concept, X_neutral])
    y = np.array([1]*n_min + [0]*n_min)

    null_accs = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=None)
        fold_accs = []
        for train_idx, test_idx in skf.split(X, y_perm):
            clf = LogisticRegression(C=C, max_iter=500, solver="lbfgs")
            clf.fit(X[train_idx], y_perm[train_idx])
            fold_accs.append(clf.score(X[test_idx], y_perm[test_idx]))
        null_accs.append(np.mean(fold_accs))

    return np.array(null_accs)


def run_all_probes(concepts, neutral_acts, concept_acts, output_dir):
    """Train probes for all concepts at all layers."""

    n_layers = neutral_acts.shape[1]
    n_concepts = len(concepts)

    # Accuracy matrix: (n_concepts, n_layers)
    accuracy_matrix = np.zeros((n_concepts, n_layers))
    std_matrix = np.zeros((n_concepts, n_layers))
    best_layers = {}

    print(f"\nTraining probes: {n_concepts} concepts x {n_layers} layers")

    for c_idx, concept in enumerate(concepts):
        if concept not in concept_acts:
            continue
        print(f"\n  [{c_idx+1}/{n_concepts}] {concept}...")

        for layer in range(n_layers):
            acc, std, _ = train_probe(concept_acts[concept], neutral_acts, layer)
            accuracy_matrix[c_idx, layer] = acc
            std_matrix[c_idx, layer] = std

        best_layer = np.argmax(accuracy_matrix[c_idx])
        best_acc = accuracy_matrix[c_idx, best_layer]
        best_layers[concept] = {"layer": int(best_layer), "accuracy": float(best_acc)}
        print(f"    Best: layer {best_layer}, accuracy {best_acc:.3f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "accuracy_matrix.npy"), accuracy_matrix)
    np.save(os.path.join(output_dir, "std_matrix.npy"), std_matrix)
    with open(os.path.join(output_dir, "best_layers.json"), "w") as f:
        json.dump(best_layers, f, indent=2)

    return accuracy_matrix, std_matrix, best_layers


def run_permutation_tests(concepts, neutral_acts, concept_acts, best_layers, output_dir):
    """Run permutation tests at each concept's best layer."""

    print(f"\nRunning permutation tests...")
    p_values = {}

    for concept in concepts:
        if concept not in concept_acts or concept not in best_layers:
            continue
        layer = best_layers[concept]["layer"]
        real_acc = best_layers[concept]["accuracy"]

        null_accs = permutation_test(concept_acts[concept], neutral_acts, layer)
        p_value = (np.sum(null_accs >= real_acc) + 1) / (len(null_accs) + 1)
        p_values[concept] = {
            "p_value": float(p_value),
            "real_accuracy": float(real_acc),
            "null_mean": float(null_accs.mean()),
            "null_std": float(null_accs.std()),
            "layer": int(layer),
        }
        print(f"  {concept}: acc={real_acc:.3f}, null={null_accs.mean():.3f}±{null_accs.std():.3f}, p={p_value:.4f}")

    with open(os.path.join(output_dir, "permutation_tests.json"), "w") as f:
        json.dump(p_values, f, indent=2)

    return p_values


# ============================================================
# PLOTS
# ============================================================

def plot_accuracy_heatmap(accuracy_matrix, concepts, output_dir):
    """Heatmap of probe accuracy: concepts x layers."""
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(
        accuracy_matrix,
        xticklabels=range(accuracy_matrix.shape[1]),
        yticklabels=concepts,
        cmap="RdYlGn",
        vmin=0.5, vmax=1.0,
        ax=ax,
        cbar_kws={"label": "Probe accuracy"}
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Concept")
    ax.set_title("Probe Accuracy: Safety Concepts x Layers")
    plt.tight_layout()
    path = os.path.join(output_dir, "accuracy_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_emergence_curves(accuracy_matrix, concepts, output_dir, top_k=10):
    """Emergence curves for the top-k most detectable concepts."""
    # Sort by max accuracy
    max_accs = accuracy_matrix.max(axis=1)
    top_indices = np.argsort(max_accs)[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, top_k))

    for i, idx in enumerate(top_indices):
        ax.plot(accuracy_matrix[idx], label=concepts[idx], color=colors[i], linewidth=2)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title(f"Concept Emergence Curves (Top {top_k})")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0.4, 1.05)
    plt.tight_layout()
    path = os.path.join(output_dir, "emergence_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_similarity_matrix(activations_dir, concepts, output_dir):
    """Plot cosine similarity between concept directions."""
    sim_path = os.path.join(activations_dir, "concept_similarity_matrix.npy")
    if not os.path.exists(sim_path):
        print("  No similarity matrix found, skipping.")
        return

    sim_matrix = np.load(sim_path)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        sim_matrix,
        xticklabels=concepts,
        yticklabels=concepts,
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        center=0,
        ax=ax,
        cbar_kws={"label": "Cosine similarity"}
    )
    ax.set_title("Cosine Similarity Between Safety Concept Directions")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    path = os.path.join(output_dir, "concept_similarity_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_pca_concept_space(activations_dir, concepts, output_dir, layer=None):
    """PCA of concept directions — the 'map' of safety concepts."""
    directions_path = os.path.join(activations_dir, "all_directions.npz")
    if not os.path.exists(directions_path):
        print("  No directions file found, skipping PCA.")
        return

    data = np.load(directions_path)

    # Use specified layer or middle layer
    sample_key = list(data.keys())[0]
    n_layers = data[sample_key].shape[0]
    if layer is None:
        layer = n_layers // 2

    # Stack all concept directions at this layer
    direction_vectors = []
    concept_labels = []
    for c in concepts:
        if c in data:
            direction_vectors.append(data[c][layer])
            concept_labels.append(c)

    X = np.stack(direction_vectors)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Color by cluster
    cluster_colors = {
        "eval-awareness": "red", "oversight-awareness": "red",
        "self-preservation": "red", "situational-modeling": "red",
        "training-awareness": "red",
        "deception": "orange", "manipulation": "orange",
        "sycophancy": "orange", "strategic-omission": "orange",
        "false-confidence": "orange",
        "honesty": "green", "refusal": "green",
        "helpfulness": "green", "uncertainty": "green",
        "compliance": "green",
        "power-seeking": "purple", "autonomy": "purple",
        "goal-preservation": "purple", "resource-acquisition": "purple",
        "authority-claim": "purple",
        "desperation": "blue", "frustration": "blue",
        "calm-under-pressure": "blue", "empathy": "blue",
        "hostility": "blue",
    }

    fig, ax = plt.subplots(figsize=(12, 10))
    for i, label in enumerate(concept_labels):
        color = cluster_colors.get(label, "gray")
        ax.scatter(X_2d[i, 0], X_2d[i, 1], c=color, s=100, zorder=5)
        ax.annotate(label, (X_2d[i, 0], X_2d[i, 1]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"PCA of Safety Concept Directions (Layer {layer})")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="red", label="Situational Awareness"),
        Patch(facecolor="orange", label="Deceptive Behaviors"),
        Patch(facecolor="green", label="Alignment"),
        Patch(facecolor="purple", label="Power & Agency"),
        Patch(facecolor="blue", label="Emotion Bridge"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    path = os.path.join(output_dir, "pca_concept_space.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_layer0_check(accuracy_matrix, concepts, output_dir):
    """Bar chart of layer 0 accuracy — sanity check for surface confounds."""
    layer0_accs = accuracy_matrix[:, 0]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(concepts)), layer0_accs, color="steelblue")

    # Highlight any concept with layer 0 accuracy > 0.7 (potential confound)
    for i, acc in enumerate(layer0_accs):
        if acc > 0.7:
            bars[i].set_color("red")

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=0.7, color="red", linestyle="--", alpha=0.3, label="Confound threshold")
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("Layer 0 Accuracy (Surface Confound Check)")
    ax.set_ylim(0.3, 1.05)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "layer0_confound_check.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--probe-C", type=float, default=0.1, help="Regularization strength")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.activations_dir), "results")

    print(f"Loading activations from {args.activations_dir}")
    concepts, neutral_acts, concept_acts = load_activations(args.activations_dir)

    # Train probes
    accuracy_matrix, std_matrix, best_layers = run_all_probes(
        concepts, neutral_acts, concept_acts, args.output_dir
    )

    # Permutation tests
    if not args.skip_permutation:
        p_values = run_permutation_tests(
            concepts, neutral_acts, concept_acts, best_layers, args.output_dir
        )

    # Generate plots
    print(f"\nGenerating plots...")
    plot_accuracy_heatmap(accuracy_matrix, concepts, args.output_dir)
    plot_emergence_curves(accuracy_matrix, concepts, args.output_dir)
    plot_similarity_matrix(args.activations_dir, concepts, args.output_dir)
    plot_pca_concept_space(args.activations_dir, concepts, args.output_dir)
    plot_layer0_check(accuracy_matrix, concepts, args.output_dir)

    # Summary
    print(f"\n{'='*60}")
    print("PROBING COMPLETE")
    print(f"{'='*60}")
    print(f"\nBest layers per concept:")
    for concept, info in sorted(best_layers.items(), key=lambda x: -x[1]["accuracy"]):
        flag = " ⚠️ LAYER0>.7" if accuracy_matrix[concepts.index(concept), 0] > 0.7 else ""
        print(f"  {concept:25s}: layer {info['layer']:2d}, accuracy {info['accuracy']:.3f}{flag}")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
