"""Train and evaluate readmission models.

- Patient-grouped, stratified hold-out test set (15%), touched once at the end
- StratifiedGroupKFold (5 folds) cross-validation on the development set
- Models: LACE baseline, logistic regression, random forest, gradient boosting
- Randomised hyperparameter search for gradient boosting
- Metrics: ROC-AUC, PR-AUC, Brier, recall/precision at the top-20% risk threshold
- Interpretation (permutation importance), risk deciles, fairness by group

Usage:
    python src/train.py --data data/processed/encounters_clean.parquet
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from clean import CATEGORICAL_FEATURES, GROUP, NUMERIC_FEATURES, TARGET

SEED = 42
# numpy + Apple Accelerate emits spurious matmul RuntimeWarnings on macOS; results are unaffected
warnings.filterwarnings("ignore", message=".*encountered in matmul")
os.environ.setdefault("PYTHONWARNINGS", "ignore::RuntimeWarning")
MIN_GAIN = 0.02  # a complex model must beat logistic regression by this much CV ROC-AUC to be chosen
TEAL, CORAL, NAVY, SLATE = "#2A9D8F", "#E76F51", "#1F3A5F", "#5C6B7A"


def preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))])
    return ColumnTransformer([("num", Pipeline(num_steps), NUMERIC_FEATURES), ("cat", cat, CATEGORICAL_FEATURES)])


def models() -> dict:
    return {
        "Logistic regression": Pipeline([("prep", preprocessor(scale=True)),
                                         ("clf", LogisticRegression(C=0.5, max_iter=2000))]),
        "Random forest": Pipeline([("prep", preprocessor(scale=False)),
                                   ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=20, max_features=0.4,
                                                                  class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))]),
        "Gradient boosting": Pipeline([("prep", preprocessor(scale=False)),
                                       ("clf", HistGradientBoostingClassifier(random_state=SEED, early_stopping=True))]),
    }


def top_k_metrics(y_true, score, frac=0.2) -> dict:
    thr = np.quantile(score, 1 - frac)
    flag = score >= thr
    tp = int((flag & (y_true == 1)).sum())
    return {"threshold": float(thr), "recall_top20": tp / int(y_true.sum()), "precision_top20": tp / int(flag.sum())}


def evaluate(y_true, score) -> dict:
    out = {"roc_auc": roc_auc_score(y_true, score), "pr_auc": average_precision_score(y_true, score)}
    if score.min() >= 0 and score.max() <= 1:
        out["brier"] = brier_score_loss(y_true, score)
    out.update(top_k_metrics(np.asarray(y_true), np.asarray(score)))
    return {k: round(float(v), 4) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/encounters_clean.parquet")
    ap.add_argument("--n-iter", type=int, default=20, help="random-search iterations for gradient boosting")
    args = ap.parse_args()
    fig_dir = Path("outputs/figures"); fig_dir.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    df = pd.read_parquet(args.data)
    X, y, groups = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET], df[GROUP]

    # ---- hold-out test set, grouped by patient --------------------------
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)  # 1/7 ~ 14-15% test
    dev_idx, test_idx = next(outer.split(X, y, groups))
    X_dev, y_dev, g_dev = X.iloc[dev_idx], y.iloc[dev_idx], groups.iloc[dev_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    assert not set(g_dev) & set(groups.iloc[test_idx]), "patient leakage between dev and test"
    print(f"dev: {len(X_dev):,} stays | test: {len(X_test):,} stays | test readmission rate {y_test.mean():.1%}")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {"LACE baseline": {"cv_roc_auc_mean": round(float(roc_auc_score(y_dev, X_dev["lace_score"])), 4), "cv_roc_auc_std": 0.0}}

    # ---- cross-validated comparison ------------------------------------
    fitted = {}
    for name, pipe in models().items():
        scores = cross_val_score(pipe, X_dev, y_dev, groups=g_dev, cv=cv, scoring="roc_auc", n_jobs=-1)
        results[name] = {"cv_roc_auc_mean": round(float(scores.mean()), 4), "cv_roc_auc_std": round(float(scores.std()), 4)}
        print(f"{name:22s} CV ROC-AUC {scores.mean():.3f} ± {scores.std():.3f}")
        fitted[name] = pipe

    # ---- tune gradient boosting ----------------------------------------
    space = {"clf__learning_rate": np.logspace(-2.3, -0.7, 20), "clf__max_leaf_nodes": [15, 31, 63],
             "clf__max_depth": [None, 4, 6, 8], "clf__min_samples_leaf": [20, 50, 100, 200],
             "clf__l2_regularization": [0.0, 0.1, 1.0, 5.0], "clf__max_iter": [200, 400]}
    search = RandomizedSearchCV(models()["Gradient boosting"], space, n_iter=args.n_iter, scoring="roc_auc",
                                cv=cv, random_state=SEED, n_jobs=-1)
    search.fit(X_dev, y_dev, groups=g_dev)
    results["Gradient boosting (tuned)"] = {"cv_roc_auc_mean": round(float(search.best_score_), 4),
                                            "best_params": {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
                                                            for k, v in search.best_params_.items()}}
    print(f"tuned gradient boosting CV ROC-AUC {search.best_score_:.3f}")

    # ---- final fit on dev, single evaluation on test --------------------
    final = {"LACE baseline": None, "Logistic regression": fitted["Logistic regression"].fit(X_dev, y_dev),
             "Random forest": fitted["Random forest"].fit(X_dev, y_dev), "Gradient boosting (tuned)": search.best_estimator_}
    test_scores = {}
    for name, m in final.items():
        s = X_test["lace_score"].values.astype(float) if m is None else m.predict_proba(X_test)[:, 1]
        test_scores[name] = s
        results[name]["test"] = evaluate(y_test.values, s)
    gain = results["Gradient boosting (tuned)"]["cv_roc_auc_mean"] - results["Logistic regression"]["cv_roc_auc_mean"]
    best = "Gradient boosting (tuned)" if gain >= MIN_GAIN else "Logistic regression"
    results["selected_model"] = {"name": best, "reason": f"tuned boosting CV gain over logistic regression = {gain:+.4f} "
                                 f"({'>=' if gain >= MIN_GAIN else '<'} {MIN_GAIN} required)"}
    print("selected:", results["selected_model"])
    joblib.dump(final[best], "models/readmission_model.joblib")

    # ---- figures -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"LACE baseline": SLATE, "Logistic regression": NAVY, "Random forest": TEAL, "Gradient boosting (tuned)": CORAL}
    for name, s in test_scores.items():
        fpr, tpr, _ = roc_curve(y_test, s); prec, rec, _ = precision_recall_curve(y_test, s)
        axes[0].plot(fpr, tpr, color=colors[name], label=f"{name} ({results[name]['test']['roc_auc']:.3f})")
        axes[1].plot(rec, prec, color=colors[name], label=f"{name} ({results[name]['test']['pr_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k:"); axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate"); axes[0].set_title("ROC curve (test set)")
    axes[1].axhline(y_test.mean(), color="k", ls=":"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision-recall curve (test set)")
    for a in axes: a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "06_roc_pr.png", dpi=150); plt.close(fig)

    s = test_scores[best]
    dec = pd.DataFrame({"y": y_test.values, "decile": pd.qcut(pd.Series(s).rank(method="first"), 10, labels=range(1, 11))})
    dec_rate = dec.groupby("decile", observed=True)["y"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(dec_rate.index.astype(int), dec_rate.values, color=[TEAL] * 8 + [CORAL] * 2)
    ax.axhline(y_test.mean() * 100, color=NAVY, ls="--")
    for d, v in dec_rate.items(): ax.text(int(d), v + 0.6, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(range(1, 11)); ax.set_xlabel("Predicted risk decile"); ax.set_ylabel("Actual readmission rate (%)")
    ax.set_title(f"Top 20% of risk scores capture {results[best]['test']['recall_top20']:.0%} of readmissions")
    fig.tight_layout(); fig.savefig(fig_dir / "07_risk_deciles.png", dpi=150); plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y_test, s, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot([0, 1], [0, 1], "k:"); ax.plot(mean_pred, frac_pos, marker="o", color=CORAL, label=best); ax.legend(fontsize=8)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed rate"); ax.set_title("Calibration (test set)")
    fig.tight_layout(); fig.savefig(fig_dir / "08_calibration.png", dpi=150); plt.close(fig)

    pi = permutation_importance(final[best], X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color=TEAL); ax.set_xlabel("Drop in ROC-AUC when shuffled"); ax.set_title(f"Top 10 drivers - {best} (permutation importance)", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "09_feature_importance.png", dpi=150); plt.close(fig)
    results["feature_importance"] = {k: round(float(v), 4) for k, v in imp.sort_values(ascending=False).items()}

    # ---- fairness: recall at the top-20% threshold by group -------------
    thr = results[best]["test"]["threshold"]
    test_df = df.iloc[test_idx].assign(score=s, flag=s >= thr)
    fair = {}
    for col in ["sex", "age_band", "insurance"]:
        g = test_df[test_df[TARGET] == 1].groupby(col, observed=True)["flag"].mean()
        fair[col] = {str(k): round(float(v), 3) for k, v in g.items()}
    results["fairness_recall_by_group"] = fair

    Path("outputs/metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v.get("test", v) for k, v in results.items() if isinstance(v, dict) and "test" in v}, indent=2))
    print("fairness (recall by group):", json.dumps(fair))


if __name__ == "__main__":
    main()
