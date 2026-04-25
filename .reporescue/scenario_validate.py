"""
Step 6 Path B — Downstream-developer scenario for Neuraxle.

Plays the role of a data scientist who:
  1. Reads the README.
  2. Wants a sklearn-compatible pipeline + Neuraxle's hyperparam search.
  3. Builds an end-to-end digit recognition workflow with manual random
     search over a HyperparameterSpace, picks the best config, retrains,
     and reports test accuracy. ~70 lines of real business logic.

No reference to the rescue tree's tests/.
"""
from __future__ import annotations

import sys
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neuraxle.base import BaseStep
from neuraxle.pipeline import Pipeline
from neuraxle.steps.sklearn import SKLearnWrapper
from neuraxle.hyperparams.distributions import Choice, RandInt, Uniform
from neuraxle.hyperparams.space import HyperparameterSpace, HyperparameterSamples


class ScalerStep(BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        self._scaler = StandardScaler()

    def fit(self, X, y=None):
        self._scaler.fit(X)
        return self

    def transform(self, X):
        return self._scaler.transform(X)


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ScalerStep(),
            SKLearnWrapper(
                LogisticRegression(),
                HyperparameterSpace(
                    {
                        "C": Uniform(0.05, 5.0),
                        "max_iter": RandInt(200, 800),
                        "solver": Choice(["lbfgs", "newton-cg"]),
                    }
                ),
            ),
        ]
    )


def evaluate(pipeline: Pipeline, X_tr, y_tr, X_va, y_va) -> float:
    pipeline = pipeline.fit(X_tr, y_tr)
    y_hat = np.asarray(pipeline.predict(X_va))
    return float((y_hat == y_va).mean())


def main() -> int:
    rng = np.random.default_rng(7)
    X, y = load_digits(return_X_y=True)
    print(f"[scenario] digits dataset: X={X.shape} y={y.shape}")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=1, stratify=y_trainval
    )
    print(f"[scenario] train={X_tr.shape} val={X_va.shape} test={X_test.shape}")

    # Manual random search over Neuraxle HyperparameterSpace.
    space = HyperparameterSpace(
        {
            "lr__C": Uniform(0.05, 5.0),
            "lr__max_iter": RandInt(200, 800),
            "lr__solver": Choice(["lbfgs", "newton-cg"]),
        }
    )
    n_trials = 6
    best_acc, best_cfg = -1.0, None
    history = []
    for trial in range(n_trials):
        np.random.seed(trial * 31 + 5)
        cfg = space.rvs().to_flat_dict()
        p = make_pipeline()
        p.set_hyperparams(
            HyperparameterSamples(
                {
                    "SKLearnWrapper_LogisticRegression": {
                        "C": cfg["lr__C"],
                        "max_iter": cfg["lr__max_iter"],
                        "solver": cfg["lr__solver"],
                    }
                }
            )
        )
        acc = evaluate(p, X_tr, y_tr, X_va, y_va)
        history.append((trial, dict(cfg), acc))
        print(f"[scenario] trial {trial}: cfg={dict(cfg)} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_cfg = acc, dict(cfg)

    assert best_cfg is not None
    print(f"[scenario] best val_acc={best_acc:.4f} cfg={best_cfg}")
    assert best_acc > 0.90, f"random search should find >0.90 on digits, got {best_acc}"

    # Retrain best config on train+val, evaluate on held-out test.
    final = make_pipeline()
    final.set_hyperparams(
        HyperparameterSamples(
            {
                "SKLearnWrapper_LogisticRegression": {
                    "C": best_cfg["lr__C"],
                    "max_iter": best_cfg["lr__max_iter"],
                    "solver": best_cfg["lr__solver"],
                }
            }
        )
    )
    final = final.fit(X_trainval, y_trainval)
    y_pred = np.asarray(final.predict(X_test))
    test_acc = float((y_pred == y_test).mean())
    print(f"[scenario] HELD-OUT test accuracy = {test_acc:.4f}")
    assert test_acc > 0.90, f"final test acc too low: {test_acc}"
    assert y_pred.shape == y_test.shape

    print("SCENARIO_USABLE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
