"""Exploratory data analysis: data-quality profile, distributions, relationships
with the outcome, correlations and time trends. Saves figures and a findings table.

Usage:
    python src/eda.py --data data/processed/encounters_clean.parquet
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu

TEAL, CORAL, NAVY = "#2A9D8F", "#E76F51", "#1F3A5F"
sns.set_theme(style="whitegrid", font_scale=0.95)


def rate_by(df, col, ax, target="readmitted_30d"):
    """Bar chart of readmission rate per category, annotated with n."""
    s = df.groupby(col, observed=True)[target].agg(["mean", "size"]).sort_values("mean")
    ax.barh(s.index.astype(str), s["mean"] * 100, color=TEAL)
    for i, (r, n) in enumerate(zip(s["mean"], s["size"])):
        ax.text(r * 100 + 0.3, i, f"{r:.1%} (n={n:,})", va="center", fontsize=8)
    ax.axvline(df[target].mean() * 100, color=CORAL, ls="--", lw=1)
    ax.set_xlabel("Readmission rate (%)"); ax.set_ylabel(""); ax.set_title(f"Readmission rate by {col}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/encounters_clean.parquet")
    ap.add_argument("--figdir", default="outputs/figures")
    args = ap.parse_args()
    fig_dir = Path(args.figdir); fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.data)
    y = "readmitted_30d"
    findings = []

    # 1. data quality
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.barh(miss.index, miss.values * 100, color=NAVY); ax.set_xlabel("% missing"); ax.set_title("Missing values after cleaning")
    fig.tight_layout(); fig.savefig(fig_dir / "01_missing.png", dpi=150); plt.close(fig)
    r_tested = df.groupby("hba1c_tested")[y].mean()
    informative = abs(r_tested.get(1, 0) - r_tested.get(0, 0)) > 0.01
    findings.append((f"HbA1c is missing for most stays; missingness {'is' if informative else 'is not'} linked to the outcome",
                     f"{miss.get('hba1c', 0):.0%} missing; readmission {r_tested.get(1, 0):.1%} if tested vs {r_tested.get(0, 0):.1%} if not",
                     "Impute and keep the hba1c_tested indicator" if informative else "Impute with median; indicator adds little here"))

    # 2. univariate
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    sns.histplot(df["age"], bins=30, ax=axes[0], color=NAVY); axes[0].set_title("Age")
    sns.histplot(df["length_of_stay"], bins=40, ax=axes[1], color=NAVY); axes[1].set_title(f"Length of stay (skew {df.length_of_stay.skew():.1f})")
    df[y].value_counts(normalize=True).sort_index().mul(100).plot.bar(ax=axes[2], color=[TEAL, CORAL])
    axes[2].set_xticklabels(["Not readmitted", "Readmitted"], rotation=0); axes[2].set_ylabel("%"); axes[2].set_title("Target balance")
    fig.tight_layout(); fig.savefig(fig_dir / "02_univariate.png", dpi=150); plt.close(fig)
    findings.append(("Target is imbalanced", f"{df[y].mean():.1%} of stays are readmissions",
                     "Use ROC-AUC / PR-AUC and recall, class weights, threshold tuning"))
    findings.append(("Length of stay is right-skewed", f"skewness {df.length_of_stay.skew():.2f}", "log-transform for linear models"))

    # 3. bivariate
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    s_dispo = rate_by(df, "discharge_disposition", axes[0, 0])
    s_cond = rate_by(df, "primary_condition", axes[0, 1])
    df["prior_adm_band"] = pd.cut(df["prior_admissions_12m"], [-1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"])
    s_prior = rate_by(df, "prior_adm_band", axes[1, 0])
    sns.boxplot(data=df, x=y, y="num_medications", ax=axes[1, 1], hue=y, palette=[TEAL, CORAL], legend=False)
    axes[1, 1].set_xticks([0, 1], ["Not readmitted", "Readmitted"]); axes[1, 1].set_xlabel(""); axes[1, 1].set_title("Number of medications by outcome")
    fig.tight_layout(); fig.savefig(fig_dir / "03_bivariate.png", dpi=150); plt.close(fig)

    chi2, p_dispo, _, _ = chi2_contingency(pd.crosstab(df["discharge_disposition"], df[y]))
    findings.append(("Discharge destination matters",
                     f"Skilled nursing {s_dispo.loc['Skilled nursing', 'mean']:.1%} vs Home {s_dispo.loc['Home', 'mean']:.1%} (chi-square p={p_dispo:.1e})",
                     "Include disposition; candidate for targeted handover"))
    findings.append(("Prior admissions are a strong signal",
                     f"0 prior: {s_prior.loc['0', 'mean']:.1%}; 3+ prior: {s_prior.loc['3+', 'mean']:.1%}",
                     "Utilisation history is a key feature"))
    findings.append(("Condition-specific risk", f"highest: {s_cond['mean'].idxmax()} ({s_cond['mean'].max():.1%}); lowest: {s_cond['mean'].idxmin()} ({s_cond['mean'].min():.1%})",
                     "Include primary condition"))
    _, p_meds = mannwhitneyu(df.loc[df[y] == 1, "num_medications"], df.loc[df[y] == 0, "num_medications"])
    findings.append(("Readmitted patients take more medications",
                     f"median {df.loc[df[y] == 1, 'num_medications'].median():.0f} vs {df.loc[df[y] == 0, 'num_medications'].median():.0f} (Mann-Whitney p={p_meds:.1e})",
                     "Medication review is an actionable lever"))
    fu = df.groupby("follow_up_7d")[y].mean()
    findings.append(("Early follow-up is associated with lower readmission",
                     f"{fu.get(1, 0):.1%} with follow-up in 7 days vs {fu.get(0, 0):.1%} without",
                     "Association only; test causally in a pilot"))

    # 4. multivariate
    cols = ["age", "length_of_stay", "charlson_index", "num_diagnoses", "num_medications",
            "prior_admissions_12m", "ed_visits_6m", "lace_score", y]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, annot_kws={"size": 7})
    ax.set_title("Spearman correlation matrix"); fig.tight_layout(); fig.savefig(fig_dir / "04_correlation.png", dpi=150); plt.close(fig)
    top = corr.loc["charlson_index", "num_diagnoses"]
    findings.append(("Charlson index and number of diagnoses overlap", f"Spearman r = {top:.2f}", "Watch multicollinearity in logistic regression"))

    # 5. time trend
    by_month = df.set_index("discharge_date").resample("MS")[y].agg(["mean", "size"])
    monthly = by_month.loc[by_month["size"] >= 100, "mean"].mul(100)  # drop sparse edge months
    fig, ax = plt.subplots(figsize=(10, 3.2))
    monthly.plot(ax=ax, color=NAVY, marker="o", ms=3); ax.set_ylabel("Readmission rate (%)"); ax.set_xlabel("")
    ax.set_title("Monthly readmission rate"); fig.tight_layout(); fig.savefig(fig_dir / "05_trend.png", dpi=150); plt.close(fig)
    win = df.groupby("winter_discharge")[y].mean()
    findings.append(("Readmissions are slightly higher in winter", f"{win.get(1, 0):.1%} in Dec-Feb vs {win.get(0, 0):.1%} other months", "Plan extra winter capacity; monitor seasonal drift"))

    out = pd.DataFrame(findings, columns=["finding", "evidence", "implication"])
    out.insert(0, "id", [f"F{i + 1}" for i in range(len(out))])
    out.to_csv("outputs/eda_findings.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
