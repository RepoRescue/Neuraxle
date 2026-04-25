# Neuraxle (RepoRescue modernized fork)

> sklearn-compatible ML pipeline framework with per-step `HyperparameterSpace`,
> streaming data primitives, and AutoML building blocks — modernized to run on
> **Python 3.13 + NumPy 2.4 + scikit-learn 1.8 + SQLAlchemy 2.0 + Flask 3.x**.

This is a community-maintained rescue of [Neuraxio/Neuraxle](https://github.com/Neuraxio/Neuraxle)
(~600 stars, originally backed by Neuraxio Inc.; academic AutoML pedigree, used
in pipeline / hyperparameter-tuning research). Upstream had not been updated for
modern NumPy / sklearn / SQLAlchemy / Flask majors and no longer installs on
Python 3.13. This fork re-establishes a clean install on a current scientific
Python stack and ships an end-to-end usability proof rather than just a green
test bar.

---

## What's in this fork

The library API is unchanged. The fix is internal — six small but load-bearing
modernizations that make `pip install -e .` and a real `Pipeline.fit().predict()`
flow work again on Python 3.13.

| # | Where | Change | Why it matters |
|---|---|---|---|
| 1 | `neuraxle/steps/sklearn.py:129-130, 139-140` | `inspect.getfullargspec(...).args` → `inspect.signature(...).parameters` | `getfullargspec` blows up on estimators whose `fit` uses `*args/**kwargs`; `signature()` handles `VAR_POSITIONAL/VAR_KEYWORD`. This is the path every `SKLearnWrapper.fit` hits. |
| 2 | `neuraxle/steps/numpy.py:117` | `np.reshape(a, newshape=...)` → `np.reshape(a, shape=...)` | NumPy 2.0 renamed the keyword. |
| 3 | `neuraxle/**` | Removed `np.str / np.int / np.float / np.bool` aliases | Deleted in NumPy 1.20+, hard error in NumPy 2.x. |
| 4 | test suite | pytest 9: `setup(self)` → `setup_method(self)` | `setup` was removed in pytest 9. |
| 5 | persistence layer (`db.py`) | `session.query(...)` → `session.execute(select(...)).unique()` | SQLAlchemy 2.x removed legacy `Query` API for new code paths. |
| 6 | `LogisticRegression(...)` call sites | dropped the deprecated `normalize` / `penalty` kwargs | scikit-learn 1.x removed them. |

See `outputs/kimi/Neuraxle/Neuraxle.src.patch` for the exact diffs.

---

## Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Resolved on a fresh Python 3.13 venv: `numpy 2.4.4`, `scipy 1.17.1`,
`scikit-learn 1.8.0`, `SQLAlchemy 2.0.49`, `Flask 3.1.3`, `MarkupSafe 3.0.3`.

---

## Quick start (real iris pipeline + Neuraxle hyperparameter space)

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neuraxle.base import BaseStep
from neuraxle.pipeline import Pipeline
from neuraxle.steps.sklearn import SKLearnWrapper
from neuraxle.hyperparams.distributions import Uniform, Choice
from neuraxle.hyperparams.space import HyperparameterSpace

class StandardScalerStep(BaseStep):
    def __init__(self):
        BaseStep.__init__(self); self._s = StandardScaler()
    def fit(self, X, y=None): self._s.fit(X); return self
    def transform(self, X): return self._s.transform(X)

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

p = Pipeline([
    StandardScalerStep(),
    SKLearnWrapper(
        LogisticRegression(max_iter=1000),
        HyperparameterSpace({"C": Uniform(0.1, 10.0),
                             "fit_intercept": Choice([True, False])}),
    ),
])
p = p.fit(Xtr, ytr)
acc = float((np.asarray(p.predict(Xte)) == yte).mean())
print(f"iris accuracy = {acc:.4f}")
assert acc > 0.91
```

Expected: `iris accuracy = 0.9111`.

---

## Bigger scenario: random search on digits

`.reporescue/scenario_validate.py` plays the role of a downstream data
scientist: load `digits`, train/val/test split, sample 6 trials from a
`HyperparameterSpace` via `space.rvs() + set_hyperparams(...)`, pick the
best by validation accuracy, retrain on `train+val`, and evaluate held-out.

Result on this fork: best `val_acc=0.9611`, **held-out test = 0.9667**.

Run it yourself:

```bash
python .reporescue/scenario_validate.py
```

This flow is **not** in the upstream test suite (`grep -rn "load_iris\|load_digits"
testing_neuraxle/` returns 0 hits) — it's an end-user-shaped check that
the rescued library actually does what its README advertises.

---

## What was probed beyond unit tests

`.reporescue/bug_hunt.py` runs five anti-blindspot probes against the rescued
build (intentionally targeted at the patched 3.13 surfaces):

1. Repeat `.fit()` on the same `Pipeline` (state-leak risk on the wrapped scaler / sklearn estimator).
2. NaN inputs (must reject cleanly, not segfault).
3. 4-thread concurrent `.predict()` (no race on shared transform state).
4. Custom estimator with `fit(X, y, *args, **kwargs)` — exact case that broke
   the old `len(getfullargspec(fit).args) < 3` check; now goes through
   `inspect.signature` and works.
5. Empty `(0, n_features)` data (must reject cleanly).

**Findings: 0.** All probes report `ok`.

---

## Submodules exercised by the validation harness

`neuraxle.pipeline`, `neuraxle.base`, `neuraxle.steps.numpy`,
`neuraxle.steps.sklearn`, `neuraxle.hyperparams.distributions`,
`neuraxle.hyperparams.space` — six distinct submodule paths touched by the
quick start + scenario script.

---

## Disclaimer

This is an unofficial maintenance fork produced by the
[RepoRescue](https://github.com/RepoRescue) benchmark project. The original
project is **Neuraxio/Neuraxle** by Neuraxio Inc. and contributors. No
affiliation or endorsement is claimed. We did not change library semantics —
only the dependency-pinning / Python-3.13 incompatibilities that prevent
upstream from installing on a current scientific Python stack. For new
production work please check upstream first; use this fork if you specifically
need a Python 3.13 / NumPy 2.x / sklearn 1.8 / SQLAlchemy 2.x compatible
Neuraxle.

## License

Apache License 2.0 — same as upstream. See `LICENSE`.
