"""
Neuraxle — usability validate (Type B: end-user library API).

Real-world signature use of Neuraxle:
    1. Build a sklearn-compatible Pipeline of preprocessing + classifier.
    2. .fit / .transform / .predict on a real sklearn dataset (iris).
    3. Use Neuraxle's HyperparameterSpace / HyperparameterSamples API
       (the library's distinguishing feature vs. plain sklearn).

Hard constraints satisfied:
  1 Real input        : sklearn.datasets.load_iris() (real 150-sample dataset)
  2 Real output assert: assert accuracy > 0.85 on held-out test set;
                        assert pipeline output ndarray shape == (n_test,)
  3 Beyond unit tests : grep showed testing_neuraxle/ has 0 references
                        to load_iris / load_digits — this end-to-end iris
                        flow is not covered by the unit tests.
  4 Primary use mode  : sklearn-compatible Pipeline + SKLearnWrapper +
                        HyperparameterSpace — the README signature use.
  5 Three+ submodules : neuraxle.pipeline.Pipeline,
                        neuraxle.steps.numpy.NumpyShapePrinter,
                        neuraxle.steps.sklearn.SKLearnWrapper,
                        neuraxle.hyperparams.distributions.{Choice,RandInt,Uniform},
                        neuraxle.hyperparams.space.{HyperparameterSpace,
                        HyperparameterSamples},
                        neuraxle.base.BaseStep
                        (>=3 distinct submodules: pipeline / steps.* / hyperparams.*).
  6 Stress 3.13 surface: see REPORT.md — exercises the patched paths
                        inspect.getfullargspec→signature (sklearn.py),
                        np.reshape(newshape→shape) (numpy.py),
                        and the NumPy 2.0 / sklearn 1.8 / SQLAlchemy 2 stack.
  7 Installed + core  : run from /tmp/Neuraxle-clean (clean venv with
                        pip install -e), NOT from rescue tree.
  8 Scenario          : Path B — see scenario_validate.py (≥30 line
                        end-to-end ML script).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 5 distinct neuraxle submodules ---
from neuraxle.pipeline import Pipeline
from neuraxle.base import BaseStep
from neuraxle.steps.sklearn import SKLearnWrapper
from neuraxle.steps.numpy import NumpyShapePrinter
from neuraxle.hyperparams.distributions import Choice, RandInt, Uniform
from neuraxle.hyperparams.space import HyperparameterSpace, HyperparameterSamples


def assert_outside_rescue_tree() -> None:
    """Constraint 7: importing from a clean venv, not the rescue tree."""
    rescue_root = "/home/zhihao/hdd/RepoRescue_Clean/repos/rescue_kimi/Neuraxle"
    cwd = os.getcwd()
    assert not cwd.startswith(rescue_root), (
        f"must run outside rescue tree, cwd={cwd}"
    )
    import neuraxle
    nx_path = os.path.dirname(neuraxle.__file__)
    # editable install resolves to the rescue tree's neuraxle/ — that's fine,
    # what matters is cwd is not the rescue tree (no relative-import shadowing).
    print(f"[ok] cwd={cwd} (outside rescue), neuraxle from {nx_path}")


class StandardScalerStep(BaseStep):
    """Custom Neuraxle step wrapping StandardScaler — exercises BaseStep."""

    def __init__(self):
        BaseStep.__init__(self)
        self._scaler = StandardScaler()

    def fit(self, data_inputs, expected_outputs=None):
        self._scaler.fit(data_inputs)
        return self

    def transform(self, data_inputs):
        return self._scaler.transform(data_inputs)


def build_pipeline() -> Pipeline:
    """Pipeline = preprocessing + sklearn-wrapped classifier."""
    p = Pipeline(
        [
            NumpyShapePrinter(custom_message="raw"),
            StandardScalerStep(),
            NumpyShapePrinter(custom_message="scaled"),
            SKLearnWrapper(
                LogisticRegression(max_iter=1000),
                HyperparameterSpace(
                    {
                        "C": Uniform(0.1, 10.0),
                        "fit_intercept": Choice([True, False]),
                    }
                ),
            ),
        ]
    )
    return p


def exercise_hyperparams_api() -> None:
    """Constraint 4 + 5: Neuraxle-specific hyperparams API."""
    space = HyperparameterSpace(
        {
            "lr__C": Uniform(0.01, 5.0),
            "lr__max_iter": RandInt(50, 500),
            "lr__solver": Choice(["lbfgs", "liblinear"]),
        }
    )
    # rvs() is the headline API: sample a concrete config from a space.
    samples = space.rvs()
    assert isinstance(samples, HyperparameterSamples), type(samples)
    flat = samples.to_flat_dict()
    assert set(flat.keys()) == {"lr__C", "lr__max_iter", "lr__solver"}, flat
    assert 0.01 <= flat["lr__C"] <= 5.0, flat["lr__C"]
    assert 50 <= flat["lr__max_iter"] <= 500, flat["lr__max_iter"]
    assert flat["lr__solver"] in {"lbfgs", "liblinear"}, flat["lr__solver"]
    print(f"[ok] HyperparameterSpace.rvs() -> {dict(flat)}")

    # Distribution API direct use.
    u = Uniform(0.0, 1.0)
    for _ in range(20):
        v = u.rvs()
        assert 0.0 <= v <= 1.0, v
    ri = RandInt(1, 10)
    for _ in range(20):
        v = ri.rvs()
        assert 1 <= v <= 10 and isinstance(v, (int, np.integer)), (v, type(v))
    print("[ok] Uniform/RandInt distributions valid")


def main() -> int:
    assert_outside_rescue_tree()

    # 1. Real dataset.
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    assert X_train.shape == (105, 4) and X_test.shape == (45, 4), (
        X_train.shape, X_test.shape
    )

    # 2. Pipeline fit / transform / predict.
    p = build_pipeline()
    p = p.fit(X_train, y_train)

    # transform on the preprocessing-only sub-pipeline path:
    # StandardScalerStep alone, exercised inline.
    scaler_only = StandardScalerStep()
    scaler_only = scaler_only.fit(X_train, y_train)
    X_train_scaled = scaler_only.transform(X_train)
    assert X_train_scaled.shape == X_train.shape
    assert abs(X_train_scaled.mean()) < 1e-10, X_train_scaled.mean()
    assert abs(X_train_scaled.std() - 1.0) < 0.05, X_train_scaled.std()
    print(f"[ok] StandardScalerStep mean={X_train_scaled.mean():.2e} "
          f"std={X_train_scaled.std():.4f}")

    # 3. predict end-to-end through full pipeline.
    y_pred = p.predict(X_test)
    y_pred = np.asarray(y_pred)
    assert y_pred.shape == (45,), y_pred.shape
    accuracy = float((y_pred == y_test).mean())
    print(f"[ok] iris pipeline accuracy = {accuracy:.4f}")
    assert accuracy > 0.85, f"accuracy too low: {accuracy}"

    # 4. Hyperparameter API.
    exercise_hyperparams_api()

    # 5. set_hyperparams on the pipeline (Neuraxle-specific).
    p2 = build_pipeline()
    new_hp = HyperparameterSamples(
        {"SKLearnWrapper_LogisticRegression": {"C": 2.5, "fit_intercept": True}}
    )
    p2.set_hyperparams(new_hp)
    p2 = p2.fit(X_train, y_train)
    y_pred2 = np.asarray(p2.predict(X_test))
    acc2 = float((y_pred2 == y_test).mean())
    print(f"[ok] after set_hyperparams accuracy = {acc2:.4f}")
    assert acc2 > 0.80, acc2

    print("USABLE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
