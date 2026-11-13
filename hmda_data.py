"""Shared data steps for every model in the project. Import from here; do not re-implement.

    applications = load_applications(pq_dir, "NC")      # every home-purchase application
    applications = add_split(applications)              # adds a train/test label to every row
    then:  approval_sample(applications)                # for the approval models
      or:  build_features(price_sample(applications))   # for the price model

The split is assigned once, on the full application table, before any filtering. Every model
therefore holds out the same loans. It depends on row order, so everyone must build the parquet
files from the same HMDA downloads with prep.py.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

YEARS = [2020, 2021, 2022, 2023, 2024]
TEST_SHARE = 0.2
SPLIT_SEED = 0
MIN_LOANS = 100                 # originations a lender needs in a year to be modelled or ranked

# TEAM DECISION (not yet confirmed): which action_taken codes count as approved or denied.
# Everything else is dropped: 4 withdrawn, 5 incomplete, 6 purchased loan, 7-8 preapprovals.
APPROVED_CODES = {1, 2}         # 1 originated, 2 approved but not accepted
DENIED_CODES = {3}

COLUMNS = [
    "activity_year", "lei", "county_code", "action_taken", "loan_type", "lien_status",
    "reverse_mortgage", "open-end_line_of_credit", "business_or_commercial_purpose",
    "loan_amount", "loan_to_value_ratio", "interest_rate", "rate_spread", "loan_term",
    "occupancy_type", "total_units", "income", "debt_to_income_ratio", "applicant_age",
    "construction_method", "tract_minority_population_percent", "tract_to_msa_income_percentage",
    "derived_race", "derived_ethnicity", "derived_sex",
]

# Model inputs. Numeric ones are standardized; categorical ones are one-hot encoded.
NUMERIC = ["log_loan_amount", "log_income", "ltv", "dti", "tract_minority", "tract_income_pct"]
CATEGORICAL = ["activity_year", "loan_type", "occupancy_type", "total_units_c", "term_c",
               "construction_method", "age", "county_code"]
# Carried in the feature table but NOT used as inputs unless a model adds them on purpose.
# Note: "age" above is also a protected characteristic; move it here for the demographic run.
DEMOGRAPHIC = ["race", "ethnicity", "sex"]

# HMDA reports DTI as exact integers from 36 to 49 and as bins elsewhere; bins get a midpoint.
DTI_BIN_MIDPOINT = {"<20%": 15, "20%-<30%": 25, "30%-<36%": 33, "50%-60%": 55, ">60%": 65}


# ------------------------------------------------------------------------------- loading
def load_applications(pq_dir, state, years=YEARS, originated_only=False):
    """Home-purchase applications for one state. `originated_only` saves memory for large states."""
    frames = []
    for year in years:
        df = pd.read_parquet(os.path.join(pq_dir, state, f"state_{state}_{year}.parquet"), columns=COLUMNS)
        frames.append(df[df.action_taken == 1] if originated_only else df)
    return pd.concat(frames, ignore_index=True)


def add_split(applications, test_share=TEST_SHARE, seed=SPLIT_SEED):
    """Label every application 'train' or 'test' (same share of each year in both)."""
    row = np.arange(len(applications))
    _, test_rows = train_test_split(row, test_size=test_share, random_state=seed,
                                    stratify=applications.activity_year)
    applications = applications.copy()
    applications["split"] = "train"
    applications.iloc[test_rows, applications.columns.get_loc("split")] = "test"
    return applications


# ---------------------------------------------------------------------- approval sample
def approval_sample(applications):
    """Applications with an underwriting decision, plus `approved` (1 or 0). See TEAM DECISION above."""
    decided = applications[applications.action_taken.isin(APPROVED_CODES | DENIED_CODES)].copy()
    decided["approved"] = decided.action_taken.isin(APPROVED_CODES).astype(int)
    return decided


# ------------------------------------------------------------------------- price sample
def clean_originations(applications):
    """Originated consumer, first-lien, closed-end, non-reverse loans with a plausible interest rate."""
    a = applications
    keep = (
        (a.action_taken == 1)
        & (a.lien_status == 1)
        & (a.reverse_mortgage == 2)
        & (a["open-end_line_of_credit"] == 2)
        & (a.business_or_commercial_purpose == 2)
        & a.interest_rate.between(0.5, 15)       # drops junk codes such as 0 and 999
        & (a.loan_amount > 0)
    )
    return a[keep].copy()


def qualifying_lender_years(clean_loans, min_loans=MIN_LOANS):
    """Set of (lei, year) pairs with at least `min_loans` cleaned originations."""
    counts = clean_loans.groupby(["lei", "activity_year"]).size()
    return set(counts[counts >= min_loans].index)


def keep_high_volume_lender_years(clean_loans, min_loans=MIN_LOANS):
    loans_in_lender_year = clean_loans.groupby(["lei", "activity_year"]).lei.transform("size")
    return clean_loans[loans_in_lender_year >= min_loans]


def price_sample(applications):
    """The loans the price model uses: cleaned originations from high-volume lender-years."""
    return keep_high_volume_lender_years(clean_originations(applications))


# ----------------------------------------------------------------------------- features
def build_features(loans, target="spread", spread_range=(-5, 10)):
    """Model table: lender id, inputs, demographics, split label, and the target if requested.

    target="spread" adds the price target and drops rows missing it; target=None keeps all rows
    (use that for approval models, which bring their own `approved` column).
    Rows missing a numeric input are dropped.
    """
    as_code = lambda s: s.astype("Int64").astype(str)          # 1.0 -> "1", missing -> "<NA>"
    dti, term = loans.debt_to_income_ratio, loans.loan_term
    f = pd.DataFrame({
        "lei": loans.lei.fillna("MISSING"),
        # numeric inputs
        "log_loan_amount": np.log(loans.loan_amount.where(loans.loan_amount > 0)),
        "log_income": np.log(loans.income.where(loans.income > 0)),      # income is in $ thousands
        "ltv": loans.loan_to_value_ratio.where(loans.loan_to_value_ratio.between(1, 200)),
        "dti": pd.to_numeric(dti, errors="coerce").fillna(dti.map(DTI_BIN_MIDPOINT)),
        "tract_minority": loans.tract_minority_population_percent,
        "tract_income_pct": loans.tract_to_msa_income_percentage,
        # categorical inputs
        "activity_year": as_code(loans.activity_year),
        "loan_type": as_code(loans.loan_type),                  # 1 conventional, 2 FHA, 3 VA, 4 USDA
        "occupancy_type": as_code(loans.occupancy_type),
        "total_units_c": as_code(loans.total_units.clip(upper=4)),
        "term_c": np.select([term == 360, term == 180, term.isna()], ["360", "180", "NA"], "other"),
        "construction_method": as_code(loans.construction_method),
        "age": loans.applicant_age.fillna("NA"),
        "county_code": loans.county_code.fillna("NA").astype(str),
        # demographics (not model inputs by default)
        "race": loans.derived_race.fillna("NA"),
        "ethnicity": loans.derived_ethnicity.fillna("NA"),
        "sex": loans.derived_sex.fillna("NA"),
    }, index=loans.index)
    f["is_gov"] = (f.loan_type != "1").astype(float)            # government-backed loan flag
    for passthrough in ["split", "approved"]:
        if passthrough in loans:
            f[passthrough] = loans[passthrough]
    required = list(NUMERIC)
    if target == "spread":
        f["spread"] = loans.rate_spread.where(loans.rate_spread.between(*spread_range))
        required.append("spread")
    return f.dropna(subset=required)
