"""Reproducible evaluation runner for RazorRisk.

RazorRisk uses one coherent synthetic dataset as the source of truth for
training and evaluating both model branches. The tabular model consumes
transaction-level behavioral features; the GNN consumes the relational
User/Device/IP/Merchant graph. The learned stacker combines predictions
from the same held-out synthetic transactions.

Usage
-----
Synthetic/project artifacts (no retraining):
    python tests/evaluate_models.py --dataset synthetic

Synthetic, retrain all models first:
    python tests/evaluate_models.py --dataset synthetic --retrain

The former public ULB/Kaggle benchmark is intentionally no longer part of
the RazorRisk evaluation contract because its anonymized PCA feature space
and lack of identity/graph fields do not represent the project's domain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.common import classification_report_dict


def _print_metrics(title: str, metrics: dict):
    print(f"\n## {title}\n")
    print("| Metric | Value |")
    print("|---|---:|")
    for key in ("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn"):
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                print(f"| {key} | {value:.6f} |")
            else:
                print(f"| {key} | {value} |")


def evaluate_synthetic(retrain: bool = False):
    if retrain:
        from ml.risk_aggregator import train_stacker
        metrics = train_stacker()
    else:
        model_dir = ROOT / "ml" / "models"
        with open(model_dir / "aggregator_eval.json") as f:
            metrics = json.load(f)

    print("# RazorRisk Synthetic Evaluation")
    print("\nSource: project synthetic dataset; held-out split produced by the current training pipeline.\n")
    print("| Model | ROC-AUC | PR-AUC | Accuracy | Balanced Acc. | Precision | Recall | F1 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, label in (("tabular_only", "Tabular / XGBoost"), ("gnn_only", "GraphSAGE / GNN"), ("stacked", "Learned stacker")):
        m = metrics[name]
        print(f"| {label} | {m['roc_auc']:.6f} | {m['pr_auc']:.6f} | {m['accuracy']:.6f} | {m['balanced_accuracy']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} |")
    print("\nConfusion matrices:")
    for name, label in (("tabular_only", "Tabular / XGBoost"), ("gnn_only", "GraphSAGE / GNN"), ("stacked", "Learned stacker")):
        _print_metrics(label, metrics[name])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic"], default="synthetic")
    ap.add_argument("--retrain", action="store_true", help="Retrain the full RazorRisk synthetic pipeline before evaluating.")
    args = ap.parse_args()
    evaluate_synthetic(args.retrain)


if __name__ == "__main__":
    main()
