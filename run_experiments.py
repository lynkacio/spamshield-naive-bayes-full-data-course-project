"""Reproduce SpamShield tuning, ablation, and held-out evaluation from the raw XLSX."""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from spamshield.data import load_xlsx_records, stratified_split_indices
from spamshield.features import build_feature_caches, FEATURE_MODES
from spamshield.metrics import classification_metrics
from spamshield.experiments import CachedExperimentRunner

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "SpamAssassin_H2_Data_FULL.xlsx"
RESULTS_DIR = ROOT / "results"
ASSETS_DIR = ROOT / "assets"
SEED = 42


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(exist_ok=True)

    records, removed_control_bytes = load_xlsx_records(DATA_PATH)
    texts = [row["text"] for row in records]
    labels = np.asarray([row["label"] for row in records], dtype=object)

    dev_idx, test_idx = stratified_split_indices(labels, 0.20, SEED)
    train_sub, val_sub = stratified_split_indices(labels[dev_idx], 0.25, SEED)
    train_idx = dev_idx[train_sub]
    val_idx = dev_idx[val_sub]

    base_counts, extras = build_feature_caches(texts)
    runner = CachedExperimentRunner(base_counts, extras, labels, train_idx)

    # With ~10x more data than the prototype, the vocabulary search is widened.
    alpha_values = [0.1, 0.5, 1, 2, 5]
    alpha_results = {}
    for alpha in alpha_values:
        alpha_results[str(alpha)], _, _ = runner.evaluate(
            alpha=alpha, max_words=2000, feature_mode="Words only", eval_idx=val_idx
        )
    best_alpha = max(alpha_values, key=lambda a: alpha_results[str(a)]["f1"])

    vocab_sizes = [500, 1000, 2000, 5000, 10000]
    vocab_results = {}
    for size in vocab_sizes:
        vocab_results[str(size)], _, _ = runner.evaluate(
            alpha=best_alpha, max_words=size, feature_mode="Words only", eval_idx=val_idx
        )
    best_vocab = max(vocab_sizes, key=lambda size: vocab_results[str(size)]["f1"])

    ablation = {}
    for feature_mode in FEATURE_MODES:
        ablation[feature_mode], _, _ = runner.evaluate(
            alpha=best_alpha,
            max_words=best_vocab,
            feature_mode=feature_mode,
            eval_idx=val_idx,
        )
    best_feature = max(ablation, key=lambda name: ablation[name]["f1"])

    test_metrics, test_confusion, final_payload = runner.evaluate(
        alpha=best_alpha,
        max_words=best_vocab,
        feature_mode=best_feature,
        eval_idx=test_idx,
    )

    majority_pred = ["ham"] * len(val_idx)
    majority_metrics = classification_metrics(labels[val_idx], majority_pred)

    output = {
        "dataset": {
            "total": len(records),
            "ham": int(np.sum(labels == "ham")),
            "spam": int(np.sum(labels == "spam")),
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
            "raw_control_bytes_sanitized_in_memory": int(removed_control_bytes),
        },
        "selection": {
            "best_alpha": best_alpha,
            "best_vocabulary_size": best_vocab,
            "best_feature_representation": best_feature,
        },
        "alpha_experiment": alpha_results,
        "vocabulary_experiment": vocab_results,
        "ablation": ablation,
        "majority_validation": majority_metrics,
        "held_out_test": test_metrics,
        "held_out_confusion": test_confusion,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    (RESULTS_DIR / "model.json").write_text(
        json.dumps({"feature_mode": best_feature, "model": final_payload}),
        encoding="utf-8",
    )

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(alpha_values, [alpha_results[str(a)]["f1"] for a in alpha_values], marker="o")
    plt.xlabel("Laplace smoothing alpha")
    plt.ylabel("Validation F1")
    plt.title("Effect of Laplace Smoothing")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "alpha_tuning.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(vocab_sizes, [vocab_results[str(v)]["f1"] for v in vocab_sizes], marker="o")
    plt.xlabel("Vocabulary size")
    plt.ylabel("Validation F1")
    plt.title("Effect of Vocabulary Size")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "vocabulary_tuning.png", dpi=180)
    plt.close()

    names = list(ablation)
    plt.figure(figsize=(9, 4.8))
    plt.bar(names, [ablation[name]["f1"] for name in names])
    plt.ylabel("Validation F1")
    plt.xlabel("Feature representation")
    plt.title("Structural Feature Ablation")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0.90, 0.98)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "feature_ablation.png", dpi=180)
    plt.close()

    print(f"Loaded {len(records)} emails; sanitized {removed_control_bytes} invalid XML control bytes in memory.")
    print(f"Split: {len(train_idx)}/{len(val_idx)}/{len(test_idx)} train/validation/test")
    print(f"Selected alpha={best_alpha}, vocab={best_vocab}, feature='{best_feature}'")
    print("Held-out test:", ", ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))
    print("Confusion:", test_confusion)


if __name__ == "__main__":
    main()
