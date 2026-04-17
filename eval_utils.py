import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             confusion_matrix, classification_report)
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import config as cfg
from dataset import SubjectAwareDataset


def predict_on_subjects(model, h5_path, subject_ids, mean, std, device="cuda", batch_size=16):
    
    model.eval()
    ds = SubjectAwareDataset(h5_path, subject_ids, mean=mean, std=std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    subj_preds = defaultdict(list)
    subj_true = defaultdict(list)

    with torch.no_grad():
        for x, y, subj_batch in loader:
            x = x.to(device)
            logits = model(x)  # (B, S, C)
            preds = logits.argmax(dim=-1).cpu().numpy()  # (B, S)
            y_np = y.numpy()

            for b in range(x.size(0)):
                sid = subj_batch[b]
                for t in range(cfg.SEQ_LEN):
                    if y_np[b, t] != -1:
                        subj_preds[sid].append(preds[b, t])
                        subj_true[sid].append(y_np[b, t])

    results = {}
    for sid in subject_ids:
        if sid in subj_preds:
            results[sid] = {
                "y_true": np.array(subj_true[sid]),
                "y_pred": np.array(subj_preds[sid]),
            }
    return results


def compute_metrics(y_true, y_pred):
    """Compute standard sleep staging metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1_per_class = f1_score(y_true, y_pred, labels=range(cfg.NUM_CLASSES),
                            average=None, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=range(cfg.NUM_CLASSES),
                        average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=range(cfg.NUM_CLASSES))
    return {
        "accuracy": acc,
        "f1_per_class": {cfg.STAGE_NAMES[i]: float(f1_per_class[i]) for i in range(cfg.NUM_CLASSES)},
        "f1_macro": f1_macro,
        "kappa": kappa,
        "confusion_matrix": cm.tolist(),
    }


def compute_fold_metrics(results):
    """Aggregate metrics from per-subject results."""
    all_true = np.concatenate([r["y_true"] for r in results.values()])
    all_pred = np.concatenate([r["y_pred"] for r in results.values()])

    overall = compute_metrics(all_true, all_pred)

    per_subject = {}
    for sid, r in results.items():
        per_subject[sid] = compute_metrics(r["y_true"], r["y_pred"])

    kappas = [per_subject[sid]["kappa"] for sid in per_subject]
    overall["per_subject_kappa_mean"] = float(np.mean(kappas))
    overall["per_subject_kappa_std"] = float(np.std(kappas))

    return overall, per_subject


def aggregate_cv_results(fold_results_list):
    """
    Aggregate results across all CV folds.

    """
    all_true = []
    all_pred = []
    all_kappas = []

    for overall, per_subj in fold_results_list:
        for sid, m in per_subj.items():
            all_kappas.append(m["kappa"])
        cm = np.array(overall["confusion_matrix"])
        if len(all_true) == 0:
            all_true = cm

    # Re-aggregate from fold data: each fold has unique test subjects
    per_class_f1s = {name: [] for name in cfg.STAGE_NAMES}
    accs = []
    kappas = []
    f1_macros = []

    for overall, _ in fold_results_list:
        accs.append(overall["accuracy"])
        kappas.append(overall["kappa"])
        f1_macros.append(overall["f1_macro"])
        for name in cfg.STAGE_NAMES:
            per_class_f1s[name].append(overall["f1_per_class"][name])

    summary = {
        "accuracy": {"mean": np.mean(accs), "std": np.std(accs)},
        "f1_macro": {"mean": np.mean(f1_macros), "std": np.std(f1_macros)},
        "kappa": {"mean": np.mean(kappas), "std": np.std(kappas)},
        "per_subject_kappa": {"mean": np.mean(all_kappas), "std": np.std(all_kappas)},
        "per_class_f1": {},
    }
    for name in cfg.STAGE_NAMES:
        summary["per_class_f1"][name] = {
            "mean": np.mean(per_class_f1s[name]),
            "std": np.std(per_class_f1s[name]),
        }

    return summary


def print_metrics(metrics, title=""):
    """Pretty-print a metrics dictionary."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  Kappa:      {metrics['kappa']:.4f}")
    print(f"  Per-class F1:")
    for name in cfg.STAGE_NAMES:
        print(f"    {name}: {metrics['f1_per_class'][name]:.4f}")
    if "per_subject_kappa_mean" in metrics:
        print(f"  Per-subject kappa: {metrics['per_subject_kappa_mean']:.4f} "
              f"\u00B1 {metrics['per_subject_kappa_std']:.4f}")


def print_cv_summary(summary):
    """Pretty-print cross-validation summary."""
    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION SUMMARY ({cfg.NUM_FOLDS}-Fold)")
    print(f"{'='*60}")
    print(f"  Accuracy:   {summary['accuracy']['mean']:.4f} \u00B1 {summary['accuracy']['std']:.4f}")
    print(f"  F1 (macro): {summary['f1_macro']['mean']:.4f} \u00B1 {summary['f1_macro']['std']:.4f}")
    print(f"  Kappa:      {summary['kappa']['mean']:.4f} \u00B1 {summary['kappa']['std']:.4f}")
    print(f"  Per-subject kappa: {summary['per_subject_kappa']['mean']:.4f} "
          f"\u00B1 {summary['per_subject_kappa']['std']:.4f}")
    print(f"  Per-class F1:")
    for name in cfg.STAGE_NAMES:
        m = summary['per_class_f1'][name]['mean']
        s = summary['per_class_f1'][name]['std']
        print(f"    {name}: {m:.4f} \u00B1 {s:.4f}")


def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None, normalize=True):
    """Plot a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if normalize:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=cfg.STAGE_NAMES, yticklabels=cfg.STAGE_NAMES, ax=ax)
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=cfg.STAGE_NAMES, yticklabels=cfg.STAGE_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(history, title="", save_path=None):
    """Plot training/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_acc"], "g-", label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_per_subject_kappa(per_subject_metrics, title="Per-Subject Kappa", save_path=None):
    """Bar chart of per-subject kappa scores."""
    subjects = sorted(per_subject_metrics.keys())
    kappas = [per_subject_metrics[s]["kappa"] for s in subjects]

    fig, ax = plt.subplots(figsize=(16, 5))
    colors = ["#2ecc71" if k > 0.6 else "#e67e22" if k > 0.4 else "#e74c3c" for k in kappas]
    ax.bar(range(len(subjects)), kappas, color=colors, edgecolor="none")
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=90, fontsize=7)
    ax.set_ylabel("Cohen's Kappa")
    ax.set_title(title)
    ax.axhline(y=np.mean(kappas), color="black", linestyle="--", linewidth=1, label=f"Mean={np.mean(kappas):.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def save_results(results_dict, path):
    """Save results to JSON."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(results_dict, default=convert))
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  [SAVE] Results saved to {path}")
