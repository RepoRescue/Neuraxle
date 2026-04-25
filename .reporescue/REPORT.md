# Neuraxle — Usability Validation

**Selected rescue**: kimi (T2 PASS)
**srconly**: kimi_srconly = FAIL (rescue depends on AI-side env install / dep edits, not source-only)
**Scenario type**: B — End-user library API (sklearn-compatible Pipeline + hyperparam space)
**Real-world use**: Neuraxle is an ML pipeline framework — compose sklearn / Tensorflow / custom steps into one fittable pipeline with per-step `HyperparameterSpace` for AutoML.

## Model selection

| Model | T2 | T2 srconly |
|---|---|---|
| sonnet | FAIL | FAIL |
| gpt-codex | FAIL | FAIL |
| **kimi** | **PASS** | FAIL |
| glm | PASS | FAIL |
| minimax | FAIL | FAIL |

Priority sonnet > gpt-codex > kimi > glm > minimax. sonnet/gpt-codex FAIL → kimi.

## Step 0: Import sanity
`repos/rescue_kimi/Neuraxle/venv-t2/bin/python -c "import neuraxle"` → OK.

## Step 4: Install + core feature (clean venv)
`python3.13 -m venv /tmp/Neuraxle-clean && pip install -e repos/rescue_kimi/Neuraxle && cd /tmp/Neuraxle-clean && python <abs>/usability_validate.py` → OK. Resolved: numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0, SQLAlchemy 2.0.49, Flask 3.1.3, MarkupSafe 3.0.3 on Python 3.13.

Pipeline `[NumpyShapePrinter, StandardScalerStep(BaseStep), NumpyShapePrinter, SKLearnWrapper(LogisticRegression, HyperparameterSpace)]`:
- iris `.fit().predict()` → accuracy **0.9111** (assert > 0.85 ✅)
- `set_hyperparams(...)` then re-fit → **0.9111** (assert > 0.80 ✅)
- `HyperparameterSpace.rvs().to_flat_dict()` → keys `{lr__C, lr__max_iter, lr__solver}`, ranges respected.

## Hard constraint 5: ≥3 submodules

| Submodule | Symbol |
|---|---|
| neuraxle.pipeline | Pipeline |
| neuraxle.base | BaseStep |
| neuraxle.steps.numpy | NumpyShapePrinter |
| neuraxle.steps.sklearn | SKLearnWrapper |
| neuraxle.hyperparams.distributions | Choice, RandInt, Uniform |
| neuraxle.hyperparams.space | HyperparameterSpace, HyperparameterSamples |

Six distinct submodule paths actually executed.

## Hard constraint 6: Py 3.13 break surface stressed

| Surface | Evidence | Validate trigger |
|---|---|---|
| `inspect.getfullargspec(...).args` → `inspect.signature(...).parameters` | `outputs/kimi/Neuraxle/Neuraxle.src.patch:129-130,139-140`; `neuraxle/steps/sklearn.py` (2 inspect.signature calls) | Every SKLearnWrapper.fit goes through here: usability_validate.py 3×, scenario_validate.py 7×, bug_hunt.py `custom_estimator_signature_path` builds an estimator with `fit(X, y, *args, **kwargs)` — the exact variadic case that broke the old `len(getfullargspec(...).args) < 3` check |
| `np.reshape(newshape=) → shape=` (NumPy 2.0 kw rename) | `Neuraxle.src.patch:117`; `neuraxle/steps/numpy.py` | NumpyShapePrinter exercises neuraxle.steps.numpy under NumPy 2.4.4 |
| `np.str/np.int/np.float/np.bool` removal | `Neuraxle.patch:245-255` | NumPy 2.4.4 install proves package imports under NumPy 2.x |
| pytest 9: `setup(self)` → `setup_method(self)` | `Neuraxle.patch:233-234` | scoped |
| sklearn 1.x param removal (normalize, penalty) | UPGRADE_REPORT.md §2 | LogisticRegression under sklearn 1.8 |
| SQLAlchemy 2.x: session.query → select + execute().unique() | `Neuraxle.patch` db.py rewrite | tangential |

Multiple distinct break surfaces hit (≥1 required ✅).

## Beyond unit tests (constraint 3)
```
$ grep -rn "load_iris\|load_digits" repos/rescue_kimi/Neuraxle/testing_neuraxle/
(0 matches)
```
End-to-end iris/digits accuracy flows are NOT in the repo's test suite.

## Step 6: Downstream / Scenario
- Path A: skipped. Neuraxle (~600 stars on Neuraxio/Neuraxle) has no star-≥100 active downstream depending on it. README directs users to write client scripts.
- Path B: `scenario_validate.py`, ~70 lines — load_digits, train/val/test split, HyperparameterSpace, 6-trial random search via `space.rvs()` + `set_hyperparams`, pick best by val_acc, retrain on train+val, held-out eval. Result: **best val_acc=0.9611, held-out test=0.9667**. ✅

## Step 7: Bug-hunt
Tried (`bug_hunt.py`):
- Repeat .fit on same Pipeline (state leak) → ok.
- NaN input → cleanly rejected by sklearn (no swallowed exc / no segfault).
- Concurrent .predict from 4 threads → all (150,), no race.
- Custom estimator with `fit(X, y, *args, **kwargs)` (variadic — exact case that broke the old getfullargspec path) → works under inspect.signature.
- Empty-data edge → cleanly rejected.

Found: **none**.

## Verdict

STATUS: USABLE

Reason: kimi rescue genuinely upgrades Neuraxle to Py 3.13 + NumPy 2.x + sklearn 1.8 + SQLAlchemy 2.x. Clean-venv `pip install -e` works, README signature use (Pipeline + HyperparameterSpace + SKLearnWrapper) trains real iris/digits classifiers to 0.91/0.97, ≥6 submodules executed, multiple 3.13 break surfaces hit (notably `inspect.getfullargspec → inspect.signature` and `np.reshape(newshape= → shape=)`), and bug-hunt including the variadic-fit case all pass.
