import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ml" / "models"


def test_synthetic_evaluation_artifacts_are_complete():
    with open(MODEL_DIR / "aggregator_eval.json") as f:
        data = json.load(f)
    assert set(data) == {"tabular_only", "gnn_only", "stacked"}
    required = {"roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn"}
    for model_name, metrics in data.items():
        assert required <= set(metrics), f"{model_name} is missing metrics"
        for key in ("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1"):
            assert 0.0 <= metrics[key] <= 1.0, f"{model_name}.{key} out of range"


def test_synthetic_evaluator_exists():
    assert (ROOT / "tests" / "evaluate_models.py").exists()
