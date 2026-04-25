"""
Step 7 — Bug-hunt. Anti-PyCG-blindspot probes for the kimi rescue.

Probes:
  1. Empty-data edge: 0 samples through the pipeline.
  2. Repeat fit on same instance (state-leak risk on _scaler / sklearn).
  3. NaN / non-finite inputs.
  4. Concurrent .predict from threads (state during transform).
  5. Stress 3.13 path: SKLearnWrapper with a custom estimator whose .fit
     has *args/**kwargs (the previously broken `inspect.getfullargspec`
     code path is now `inspect.signature`).
"""
from __future__ import annotations

import threading
import traceback

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

from neuraxle.pipeline import Pipeline
from neuraxle.steps.sklearn import SKLearnWrapper
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.hyperparams.distributions import Uniform


REPORT = []


def probe(name):
    def deco(fn):
        def wrapped():
            try:
                fn()
                REPORT.append((name, "ok", ""))
                print(f"[probe ok] {name}")
            except Exception as e:
                REPORT.append((name, "FOUND", f"{type(e).__name__}: {e}"))
                print(f"[probe FOUND] {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
        return wrapped
    return deco


@probe("repeat_fit_state_leak")
def probe_repeat_fit():
    X, y = load_iris(return_X_y=True)
    p = Pipeline([SKLearnWrapper(LogisticRegression(max_iter=400))])
    for _ in range(3):
        p = p.fit(X, y)
    yp = np.asarray(p.predict(X))
    assert (yp == y).mean() > 0.9


@probe("nan_input")
def probe_nan():
    X, y = load_iris(return_X_y=True)
    Xn = X.copy()
    Xn[0, 0] = float("nan")
    p = Pipeline([SKLearnWrapper(LogisticRegression(max_iter=400))])
    raised = False
    try:
        p.fit(Xn, y)
    except Exception:
        raised = True
    # We don't claim NaN should succeed; we just want a clean exception, not segfault.
    assert raised, "expected sklearn to reject NaN, but it accepted"


@probe("concurrent_predict")
def probe_concurrent_predict():
    X, y = load_iris(return_X_y=True)
    p = Pipeline([SKLearnWrapper(LogisticRegression(max_iter=400))]).fit(X, y)

    results = []
    errs = []

    def worker():
        try:
            results.append(np.asarray(p.predict(X)))
        except Exception as e:
            errs.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errs, errs
    assert all(r.shape == (150,) for r in results)


@probe("custom_estimator_signature_path")
def probe_custom_estimator():
    """Triggers the patched inspect.signature path explicitly."""

    class WeirdClf(BaseEstimator, ClassifierMixin):
        # *args/**kwargs would have killed the old getfullargspec(...).args
        # length check; signature() handles VAR_POSITIONAL/VAR_KEYWORD fine.
        def fit(self, X, y, *args, **kwargs):
            self._mean = np.asarray(X).mean(axis=0)
            self.classes_ = np.unique(y)
            return self

        def predict(self, X, *args, **kwargs):
            return np.full(np.asarray(X).shape[0], self.classes_[0])

    X, y = load_iris(return_X_y=True)
    w = SKLearnWrapper(
        WeirdClf(),
        HyperparameterSpace({"C": Uniform(0.1, 1.0)}),  # ignored by WeirdClf
    )
    w = w.fit(X, y)
    out = np.asarray(w.transform(X))
    assert out.shape == (150,)


@probe("empty_data")
def probe_empty():
    X = np.zeros((0, 4), dtype=np.float64)
    y = np.zeros((0,), dtype=np.int64)
    p = Pipeline([SKLearnWrapper(LogisticRegression(max_iter=400))])
    raised = False
    try:
        p.fit(X, y)
    except Exception:
        raised = True
    # sklearn rejects empty; that is correct behavior, just must not crash uncleanly.
    assert raised


def main() -> int:
    probe_repeat_fit()
    probe_nan()
    probe_concurrent_predict()
    probe_custom_estimator()
    probe_empty()
    print()
    print("=== bug-hunt summary ===")
    found = [r for r in REPORT if r[1] == "FOUND"]
    for name, status, msg in REPORT:
        print(f"  {status:6}  {name}  {msg}")
    print(f"FOUND_BUGS: {len(found)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
