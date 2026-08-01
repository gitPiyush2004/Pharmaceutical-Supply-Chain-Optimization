#!/usr/bin/env python
"""
Train and persist both platform models.

1. **Drug classification** - patient-level prescribing model on the Kaggle
   ``drug200`` dataset.
2. **Batch risk classification** - stability risk tier from batch telemetry.

Each family trains a decision tree, a random forest and an XGBoost model under
identical cross-validation, selects the winner on cross-validated macro F1, and
writes the fitted pipeline plus full evaluation metadata to ``models/``.

Usage
-----
    python scripts/train_models.py                    # tune and train both
    python scripts/train_models.py --no-tune          # defaults only (fast)
    python scripts/train_models.py --model drug       # one family only
    python scripts/train_models.py --model batch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config  # noqa: E402
from src.data import loader  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.ml.train import (save_artifacts, train_all,  # noqa: E402
                          train_batch_risk_classifier, train_drug_classifier)

log = get_logger("scripts.train_models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip grid search and use default hyper-parameters.")
    parser.add_argument("--model", choices=["drug", "batch", "both"], default="both",
                        help="Which model family to train (default: both).")
    return parser.parse_args()


def report(name: str, result: dict) -> None:
    """Print the selection evidence and headline metrics for one model family."""
    metrics = result["test_metrics"]
    print("\n" + "=" * 74)
    print(f"{name.upper().replace('_', ' ')}")
    print("=" * 74)
    print(result["comparison"].to_string(index=False))
    print("-" * 74)
    print(f"  Selected            {result['best_model_name']}")
    print(f"  Test accuracy       {metrics['accuracy']:.4f}")
    print(f"  Macro F1            {metrics['f1_macro']:.4f}")
    print(f"  Weighted F1         {metrics['f1_weighted']:.4f}")
    if result.get("roc_auc_ovr") is not None:
        print(f"  ROC AUC (OvR)       {result['roc_auc_ovr']:.4f}")
    print(f"  Train / test rows   {result['n_train']:,} / {result['n_test']:,}")
    print(f"  Best params         {result['best_params']}")
    print("\n  Top features:")
    for _, row in result["feature_importance"].head(5).iterrows():
        print(f"    {row['feature']:<28} {row['importance']:.4f}")
    print("=" * 74)


def main() -> int:
    args = parse_args()
    cfg = get_config()
    tune = not args.no_tune
    started = time.perf_counter()

    print(f"\n{cfg.project.name} v{cfg.project.version}")
    print(f"Training models (tune={tune}, seed={cfg.project.random_seed})\n")

    # Training reads the supply chain data, so make sure it exists first.
    loader.ensure_datasets()

    if args.model == "both":
        results = train_all(tune=tune)
    elif args.model == "drug":
        result = train_drug_classifier(tune=tune)
        save_artifacts(result, "drug_classification")
        results = {"drug_classification": result}
    else:
        result = train_batch_risk_classifier(tune=tune)
        save_artifacts(result, "batch_risk")
        results = {"batch_risk": result}

    for name, result in results.items():
        report(name, result)

    models_dir = Path(cfg.paths.models)
    elapsed = time.perf_counter() - started
    print(f"\nTrained {len(results)} model family(ies) in {elapsed:.1f}s")
    print(f"Artefacts written to {models_dir}/")
    print("Next: streamlit run app/Home.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
