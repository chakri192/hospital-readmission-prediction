"""Generate a synthetic, de-identified hospital encounters dataset.

No real patient data is used. The generator mimics the structure and the
messiness of real EHR extracts (missing labs, outliers, duplicates, label
inconsistencies) so the cleaning and EDA steps have something to do.

Usage:
    python src/generate_data.py --patients 20000 --out data/raw/encounters.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DISPOSITIONS = ["Home", "Home health", "Skilled nursing", "Rehab"]
ADMISSION_TYPES = ["Emergency", "Urgent", "Elective"]
INSURANCE = ["Medicare", "Medicaid", "Private", "Self-pay"]
CONDITIONS = ["Heart failure", "COPD", "Pneumonia", "Diabetes", "Hip/knee replacement", "Other"]


def generate(n_patients: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- patients -------------------------------------------------------
    pid = np.arange(1, n_patients + 1)
    age = np.clip(rng.normal(64, 16, n_patients), 18, 100).round()
    sex = rng.choice(["F", "M"], n_patients)
    insurance = np.where(age >= 65, "Medicare",
                         rng.choice(INSURANCE[1:], n_patients, p=[0.3, 0.6, 0.1]))
    frailty = rng.gamma(2.0, 0.5, n_patients) + (age - 18) / 60  # latent illness burden

    # 1-4 stays per patient over three years, more for frail patients
    n_stays = 1 + rng.poisson(np.clip(frailty * 0.5, 0, 3))
    rows = np.repeat(np.arange(n_patients), n_stays)
    n = len(rows)

    df = pd.DataFrame({
        "patient_id": pid[rows],
        "age": age[rows],
        "sex": sex[rows],
        "insurance": insurance[rows],
    })
    f = frailty[rows] + rng.normal(0, 0.3, n)

    start = np.datetime64("2023-01-01")
    df["admit_date"] = start + rng.integers(0, 3 * 365 - 60, n).astype("timedelta64[D]")
    df["primary_condition"] = rng.choice(CONDITIONS, n, p=[0.14, 0.12, 0.14, 0.12, 0.10, 0.38])
    df["admission_type"] = rng.choice(ADMISSION_TYPES, n, p=[0.62, 0.20, 0.18])
    elective = (df["admission_type"] == "Elective") | (df["primary_condition"] == "Hip/knee replacement")

    df["charlson_index"] = np.clip(rng.poisson(np.clip(f * 1.4, 0.1, None)), 0, 15)
    df["num_diagnoses"] = np.clip(df["charlson_index"] + rng.poisson(4, n), 1, 25)
    df["num_medications"] = np.clip(rng.poisson(3 + f * 2.5), 0, 40)
    df["prior_admissions_12m"] = rng.poisson(np.clip(f * 0.6 - 0.3, 0.05, None))
    df["ed_visits_6m"] = rng.poisson(np.clip(f * 0.5, 0.05, None))
    df["length_of_stay"] = np.round(rng.gamma(1.8, 1.5 + f * 0.8) + 1, 1)
    df["icu_stay"] = (rng.random(n) < np.clip(0.04 + f * 0.04, 0, 0.5)).astype(int)

    dispo_p = np.column_stack([
        np.clip(0.75 - f * 0.12, 0.2, 0.9), np.full(n, 0.12),
        np.clip(0.03 + f * 0.07, 0.01, 0.45), np.clip(0.02 + f * 0.03, 0.01, 0.2),
    ])
    dispo_p /= dispo_p.sum(axis=1, keepdims=True)
    df["discharge_disposition"] = [DISPOSITIONS[i] for i in (dispo_p.cumsum(1) > rng.random((n, 1))).argmax(1)]

    df["discharge_date"] = df["admit_date"] + pd.to_timedelta(np.ceil(df["length_of_stay"]), unit="D")
    df["weekend_discharge"] = (df["discharge_date"].dt.dayofweek >= 5).astype(int)
    df["follow_up_7d"] = (rng.random(n) < np.where(elective, 0.8, 0.5)).astype(int)

    # labs at discharge
    df["sodium"] = np.round(rng.normal(139, 3.5, n) - (rng.random(n) < 0.08) * rng.uniform(5, 12, n), 0)
    df["hemoglobin"] = np.round(rng.normal(12.5, 1.8, n) - f * 0.4, 1)
    diabetic = df["primary_condition"].eq("Diabetes") | (rng.random(n) < 0.18)
    df["hba1c"] = np.where(diabetic, np.round(rng.normal(7.6, 1.4, n), 1), np.round(rng.normal(5.5, 0.5, n), 1))

    # --- outcome: unplanned readmission within 30 days -------------------
    cond_effect = df["primary_condition"].map({"Heart failure": 0.55, "COPD": 0.45, "Pneumonia": 0.2,
                                               "Diabetes": 0.15, "Hip/knee replacement": -0.9, "Other": 0.0})
    dispo_effect = df["discharge_disposition"].map({"Home": 0.0, "Home health": 0.3, "Skilled nursing": 0.75, "Rehab": 0.45})
    logit = (-3.95
             + 0.42 * np.log1p(df["prior_admissions_12m"]) * 2
             + 0.13 * df["charlson_index"]
             + 0.20 * np.log1p(df["length_of_stay"])
             + 0.035 * df["num_medications"]
             + 0.18 * df["ed_visits_6m"]
             + 0.012 * (df["age"] - 60).clip(lower=0)
             + cond_effect + dispo_effect
             - 0.45 * df["follow_up_7d"]
             + 0.15 * df["weekend_discharge"]
             + 0.35 * (df["sodium"] < 133)
             + 0.25 * (df["hemoglobin"] < 10)
             + 0.25 * ((df["num_medications"] >= 10) & (df["age"] >= 75))   # interaction
             + 0.2 * (df["admit_date"].dt.month.isin([12, 1, 2])))          # winter
    df["readmitted_30d"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    # --- realistic data-quality problems ---------------------------------
    hba1c_tested = rng.random(n) < np.where(diabetic, 0.75, 0.12)
    df.loc[~hba1c_tested, "hba1c"] = np.nan                    # informative missingness (MNAR)
    df.loc[rng.random(n) < 0.04, "sodium"] = np.nan             # random missingness (MCAR)
    df.loc[rng.random(n) < 0.03, "hemoglobin"] = np.nan
    df.loc[rng.random(n) < 0.06, "insurance"] = np.nan
    df.loc[rng.choice(n, 25, replace=False), "length_of_stay"] = rng.uniform(60, 180, 25).round(1)  # extreme stays
    df.loc[rng.choice(n, 8, replace=False), "age"] = 999                                           # entry errors
    sex_map = {"F": ["F", "Female", "female"], "M": ["M", "Male", "male"]}
    noisy = rng.random(n) < 0.05
    df.loc[noisy, "sex"] = [rng.choice(sex_map[s]) for s in df.loc[noisy, "sex"]]

    df = df.sort_values(["admit_date", "patient_id"]).reset_index(drop=True)
    df.insert(0, "encounter_id", np.arange(100001, 100001 + len(df)))
    dupes = df.sample(60, random_state=seed)                   # duplicated extract rows
    df = pd.concat([df, dupes]).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/raw/encounters.csv")
    args = ap.parse_args()
    df = generate(args.patients, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} encounters for {df.patient_id.nunique():,} patients -> {args.out}")
    print(f"readmission rate: {df.readmitted_30d.mean():.1%}")


if __name__ == "__main__":
    main()
