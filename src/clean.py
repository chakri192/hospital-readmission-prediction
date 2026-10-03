"""Data cleaning and feature engineering for the readmission dataset.

Implements the cleaning plan: validation rules, de-duplication, label
standardisation, outlier treatment and engineered features. Every step is
recorded in a cleaning log so the report can state exactly what changed.

Usage:
    python src/clean.py --raw data/raw/encounters.csv --out data/processed/encounters_clean.parquet
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "age", "length_of_stay", "num_diagnoses", "charlson_index", "num_medications",
    "prior_admissions_12m", "ed_visits_6m", "sodium", "hemoglobin", "hba1c",
    "icu_stay", "weekend_discharge", "follow_up_7d", "polypharmacy", "abnormal_sodium",
    "low_hemoglobin", "hba1c_tested", "winter_discharge", "lace_score",
]
CATEGORICAL_FEATURES = ["sex", "insurance", "admission_type", "discharge_disposition", "primary_condition"]
TARGET = "readmitted_30d"
GROUP = "patient_id"


def lace_score(df: pd.DataFrame) -> pd.Series:
    """LACE index: Length of stay, Acuity, Comorbidity, ED visits (van Walraven 2010)."""
    los = df["length_of_stay"]
    l_pts = np.select([los < 1, los <= 1, los <= 2, los <= 3, los <= 6, los <= 13], [0, 1, 2, 3, 4, 5], 7)
    a_pts = np.where(df["admission_type"].eq("Emergency"), 3, 0)
    c = df["charlson_index"]
    c_pts = np.select([c == 0, c == 1, c == 2, c == 3], [0, 1, 2, 3], 5)
    e_pts = np.clip(df["ed_visits_6m"], 0, 4)
    return pd.Series(l_pts + a_pts + c_pts + e_pts, index=df.index)


def clean(raw: pd.DataFrame):
    log = []
    df = raw.copy()

    def note(step, detail, rows=None):
        log.append({"step": step, "detail": detail, "rows_affected": int(rows) if rows is not None else None})

    n0 = len(df)
    df = df.drop_duplicates(subset="encounter_id")
    note("duplicates", "dropped duplicate encounter_id rows", n0 - len(df))

    for c in ["admit_date", "discharge_date"]:
        df[c] = pd.to_datetime(df[c])

    bad_age = ~df["age"].between(18, 110)
    df.loc[bad_age, "age"] = np.nan
    note("validation", "age outside 18-110 set to missing then imputed with median", bad_age.sum())
    df["age"] = df["age"].fillna(df["age"].median())

    bad_dates = df["discharge_date"] < df["admit_date"]
    df = df[~bad_dates]
    note("validation", "dropped rows with discharge before admission", bad_dates.sum())

    before = df["sex"].copy()
    df["sex"] = df["sex"].str.strip().str[0].str.upper()
    note("standardise", "sex labels mapped to F/M", (before != df["sex"]).sum())

    df["insurance"] = df["insurance"].fillna("Unknown")
    note("missing", "missing insurance kept as 'Unknown' category", (df["insurance"] == "Unknown").sum())

    cap = df["length_of_stay"].quantile(0.99)
    over = df["length_of_stay"] > cap
    df["length_of_stay"] = df["length_of_stay"].clip(upper=cap)
    note("outliers", f"length_of_stay winsorised at 99th percentile ({cap:.1f} days)", over.sum())

    # engineered features (all known at discharge)
    df["hba1c_tested"] = df["hba1c"].notna().astype(int)
    df["polypharmacy"] = (df["num_medications"] >= 5).astype(int)
    df["abnormal_sodium"] = ((df["sodium"] < 135) | (df["sodium"] > 145)).astype(int)
    df["low_hemoglobin"] = (df["hemoglobin"] < 10).astype(int)
    df["winter_discharge"] = df["discharge_date"].dt.month.isin([12, 1, 2]).astype(int)
    df["lace_score"] = lace_score(df)
    df["age_band"] = pd.cut(df["age"], [17, 40, 65, 75, 120], labels=["18-39", "40-64", "65-74", "75+"])
    note("features", "added hba1c_tested, polypharmacy, abnormal_sodium, low_hemoglobin, winter_discharge, lace_score, age_band")

    # sodium, hemoglobin and hba1c stay NaN here: they are imputed inside the model pipeline
    return df.reset_index(drop=True), log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/encounters.csv")
    ap.add_argument("--out", default="data/processed/encounters_clean.parquet")
    ap.add_argument("--log", default="outputs/cleaning_log.json")
    args = ap.parse_args()
    df, log = clean(pd.read_csv(args.raw))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    Path(args.log).write_text(json.dumps(log, indent=2))
    for entry in log:
        print(f"[{entry['step']}] {entry['detail']}" + (f" ({entry['rows_affected']} rows)" if entry["rows_affected"] is not None else ""))
    print(f"clean rows: {len(df):,} -> {args.out}")


if __name__ == "__main__":
    main()
