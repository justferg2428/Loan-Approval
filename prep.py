"""Extract home-purchase rows and needed columns from HMDA state-year CSVs into parquet.

Usage: python3 -I prep.py <csv_dir> <out_dir>
"""
import sys
import glob
import os
import pandas as pd

KEEP = [
    "activity_year", "lei", "state_code", "county_code", "census_tract",
    "derived_loan_product_type", "derived_dwelling_category",
    "derived_ethnicity", "derived_race", "derived_sex",
    "action_taken", "preapproval", "loan_type", "loan_purpose", "lien_status",
    "reverse_mortgage", "open-end_line_of_credit", "business_or_commercial_purpose",
    "loan_amount", "loan_to_value_ratio", "interest_rate", "rate_spread",
    "total_loan_costs", "loan_term", "property_value", "occupancy_type",
    "total_units", "income", "debt_to_income_ratio", "applicant_age",
    "conforming_loan_limit", "construction_method",
    "tract_population", "tract_minority_population_percent",
    "tract_to_msa_income_percentage", "ffiec_msa_md_median_family_income",
]

NUMERIC = [
    "activity_year", "action_taken", "preapproval", "loan_type", "loan_purpose",
    "lien_status", "reverse_mortgage", "open-end_line_of_credit",
    "business_or_commercial_purpose", "loan_amount", "loan_to_value_ratio",
    "interest_rate", "rate_spread", "total_loan_costs", "loan_term",
    "property_value", "occupancy_type", "total_units", "income",
    "construction_method", "tract_population", "tract_minority_population_percent",
    "tract_to_msa_income_percentage", "ffiec_msa_md_median_family_income",
]


def main(csv_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
        parts = []
        n_all = 0
        for chunk in pd.read_csv(path, usecols=KEEP, dtype=str, chunksize=400_000,
                                 na_values=["NA"], keep_default_na=False):
            n_all += len(chunk)
            chunk = chunk[chunk["loan_purpose"] == "1"]
            parts.append(chunk)
        df = pd.concat(parts, ignore_index=True)
        for c in NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce")  # 'Exempt' -> NaN
        base = os.path.basename(path).replace(" ", "_").replace(".csv", ".parquet")
        df.to_parquet(os.path.join(out_dir, base), index=False)
        print(f"{base}: {n_all:,} rows read -> {len(df):,} home-purchase rows", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
