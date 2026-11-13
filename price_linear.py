"""Price model with lender-specific slopes (partial pooling), fit by ridge regression.

    spread = year effect + pooled effect of loan features      (same for every lender)
           + a[lender]                                         (lender's price level)
           + b[lender] . z(x)                                  (how that lender reacts to this loan)

a and b are shrunk toward zero, so a lender with little data falls back to the pooled model.
For one applicant the first line is identical across lenders, so lenders are ranked by
    score = a[lender] + b[lender] . z(x)        (lower = cheaper).

Steps:  1 build the NC sample   2 tune shrinkage   3 fit and test (NC, then SC/TX)
        4 does the best lender change by applicant?   5 are lender scores stable?

Train and test rows come from the shared split in hmda_data.py.

Run:  python src/price_linear.py --pq data/pq --out outputs
"""
import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hmda_data import (CATEGORICAL, MIN_LOANS, NUMERIC, add_split, build_features, clean_originations,
                       load_applications, price_sample, qualifying_lender_years)

# ----------------------------------------------------------------------------------------
# Settings: every choice that moves the results lives here.
# ----------------------------------------------------------------------------------------
# (Years, the 100-loan lender rule and the train/test split are shared settings in hmda_data.py.)
SLOPE_FEATURES = ["log_loan_amount", "ltv", "dti", "log_income", "is_gov"]
ALPHA = 10.0                    # ridge penalty on the pooled coefficients
INTERCEPT_SCALES = [0.5, 1, 2, 4]       # larger scale = less shrinkage of a[lender]
SLOPE_SCALES = [0.25, 0.5, 1, 2, 4]     # larger scale = less shrinkage of b[lender]
MIN_BUCKET_LOANS = 30           # loans a lender needs in an applicant's bucket to be recommended
OOD_STATES = {"SC": 1.0, "TX": 0.15}    # test-only states and the share of rows used
PROFILES = {                    # example applicants (income in $ thousands, as in HMDA)
    "A_conventional_250k_LTV80": dict(amount=250_000, ltv=80.0, dti=36, income=90, gov=0),
    "B_FHA_180k_LTV96":          dict(amount=180_000, ltv=96.5, dti=45, income=60, gov=1),
    "C_conventional_600k_LTV70": dict(amount=600_000, ltv=70.0, dti=30, income=220, gov=0),
}


# ----------------------------------------------------------------------------------------
# Design matrix: X = [ pooled features | lender dummies | lender dummies x slope features ]
# ----------------------------------------------------------------------------------------
class LenderDesign:
    """Learns encodings on the training rows, then turns any table into the sparse matrix X."""

    def __init__(self, train):
        self.pooled = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), CATEGORICAL),
        ]).fit(train)
        self.lender_onehot = OneHotEncoder(handle_unknown="ignore").fit(train[["lei"]])
        self.lenders = self.lender_onehot.categories_[0]
        self.slope_mean = train[SLOPE_FEATURES].mean().values
        self.slope_sd = train[SLOPE_FEATURES].std().values

        n_pooled = self.pooled.transform(train.iloc[:1]).shape[1]
        n_lenders, n_slopes = len(self.lenders), len(SLOPE_FEATURES)
        self.pooled_cols = np.arange(n_pooled)
        self.intercept_cols = n_pooled + np.arange(n_lenders)
        self.slope_cols = n_pooled + n_lenders + np.arange(n_lenders * n_slopes)

    def z(self, df):
        """Standardized slope features, one row per loan."""
        return (df[SLOPE_FEATURES].values - self.slope_mean) / self.slope_sd

    def transform(self, df):
        pooled = sp.csr_matrix(self.pooled.transform(df))
        lender = self.lender_onehot.transform(df[["lei"]]).tocsr()
        z = self.z(df)
        slopes = [lender.multiply(z[:, [k]]) for k in range(z.shape[1])]   # dummy x feature k
        return sp.hstack([pooled, lender] + slopes).tocsr()


# ----------------------------------------------------------------------------------------
# Ridge solver. X has only ~1,600 columns, so we solve the normal equations exactly:
#     (X'X + diag(penalty)) w = X'y        (after centering, so the constant is unpenalized)
# X'X is computed once per dataset; trying another penalty is then a millisecond solve.
# ----------------------------------------------------------------------------------------
@dataclass
class Gram:
    xtx: np.ndarray     # centered X'X
    xty: np.ndarray     # centered X'y
    x_mean: np.ndarray
    y_mean: float

    @classmethod
    def from_data(cls, X, y):
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        x_mean = np.asarray(X.mean(axis=0)).ravel()
        xtx = (X.T @ X).toarray() - n * np.outer(x_mean, x_mean)
        xty = X.T @ y - n * x_mean * y.mean()
        return cls(xtx, xty, x_mean, y.mean())


@dataclass
class PriceModel:
    design: LenderDesign
    weights: np.ndarray         # one weight per column of X (zeros for slopes if not used)
    constant: float

    def predict(self, X):
        return X @ self.weights + self.constant

    @property
    def lender_intercepts(self):                       # a[lender]
        return self.weights[self.design.intercept_cols]

    @property
    def lender_slopes(self):                           # b[lender, feature]
        n_lenders = len(self.design.lenders)
        return self.weights[self.design.slope_cols].reshape(len(SLOPE_FEATURES), n_lenders).T

    def lender_scores(self, z):
        """Lender part of the price for applicants z: rows = applicants, columns = lenders."""
        return self.lender_intercepts[None, :] + np.atleast_2d(z) @ self.lender_slopes.T


def model_columns(design, lender_intercepts=True, lender_slopes=False):
    """Column numbers of X used by a model: pooled only, + lender intercepts, + lender slopes."""
    cols = [design.pooled_cols]
    if lender_intercepts:
        cols.append(design.intercept_cols)
    if lender_slopes:
        cols.append(design.slope_cols)
    return np.concatenate(cols)


def fit_ridge(design, gram, intercept_scale, slope_scale=None):
    """slope_scale=None fits the intercept-only model (lender slopes fixed at zero)."""
    penalty = np.full(len(gram.xty), ALPHA)
    penalty[design.intercept_cols] = ALPHA / intercept_scale ** 2
    if slope_scale is not None:
        penalty[design.slope_cols] = ALPHA / slope_scale ** 2
    use = model_columns(design, lender_slopes=slope_scale is not None)

    weights = np.zeros(len(gram.xty))
    lhs = gram.xtx[np.ix_(use, use)] + np.diag(penalty[use])
    weights[use] = scipy.linalg.solve(lhs, gram.xty[use], assume_a="pos")
    return PriceModel(design, weights, gram.y_mean - gram.x_mean @ weights)


def fit_ols(design, gram, lender_intercepts=True, lender_slopes=False):
    """Ordinary least squares: the same model with no penalty.

    One-hot columns are collinear with the constant, so we use a pseudo-inverse. Predictions
    and differences between categories are unaffected by that choice.
    """
    use = model_columns(design, lender_intercepts, lender_slopes)
    weights = np.zeros(len(gram.xty))
    weights[use] = scipy.linalg.pinvh(gram.xtx[np.ix_(use, use)]) @ gram.xty[use]
    return PriceModel(design, weights, gram.y_mean - gram.x_mean @ weights)


def clustered_covariance(X, y, model, gram, use, cluster_onehot):
    """Covariance of OLS weights (columns `use`), robust to correlated errors within a lender.

    Sandwich formula: inv(X'X) * [sum over lenders of s_g s_g'] * inv(X'X), where s_g is
    lender g's sum of (centered x) * residual.
    """
    residual = np.asarray(y) - model.predict(X)
    per_lender = (cluster_onehot.T @ X.multiply(residual[:, None])).toarray()
    per_lender -= np.outer(cluster_onehot.T @ residual, gram.x_mean)      # centering
    scores = per_lender[:, use]
    bread = scipy.linalg.pinvh(gram.xtx[np.ix_(use, use)])
    n, groups = X.shape[0], cluster_onehot.shape[1]
    rank = np.linalg.matrix_rank(gram.xtx[np.ix_(use, use)]) + 1          # +1 for the constant
    correction = groups / (groups - 1) * (n - 1) / (n - rank)
    return correction * bread @ (scores.T @ scores) @ bread


# ----------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------
def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - pred) ** 2)))


def score_predictions(y, pred, year_mean_pred):
    """Fit metrics, plus the share of within-year variance explained (vs predicting the year mean)."""
    y = np.asarray(y)
    model_rmse, year_rmse = rmse(y, pred), rmse(y, year_mean_pred)
    return dict(n=len(y), rmse=model_rmse, mae=float(np.mean(np.abs(y - pred))),
                r2=float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
                year_only_rmse=year_rmse,
                within_year_var_explained=1 - model_rmse ** 2 / year_rmse ** 2)


# ----------------------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------------------
def build_nc_sample(pq_dir):
    """NC price sample with the shared train/test label on every row."""
    loans = price_sample(add_split(load_applications(pq_dir, "NC")))
    lenders_active_2024 = set(loans[loans.activity_year == 2024].lei.dropna())
    return build_features(loans), lenders_active_2024


def tune_shrinkage(train):
    """Pick the two shrinkage scales on a validation slice of the training rows."""
    fit_part, val_part = train_test_split(train, test_size=0.15, random_state=1)
    design = LenderDesign(fit_part)
    gram = Gram.from_data(design.transform(fit_part), fit_part.spread)
    X_val = design.transform(val_part)
    val_rmse = lambda a_scale, b_scale: rmse(val_part.spread, fit_ridge(design, gram, a_scale, b_scale).predict(X_val))

    intercept_curve = {s: val_rmse(s, None) for s in INTERCEPT_SCALES}
    best_a = min(intercept_curve, key=intercept_curve.get)
    slope_curve = {s: val_rmse(best_a, s) for s in SLOPE_SCALES}
    best_b = min(slope_curve, key=slope_curve.get)
    return best_a, best_b, dict(intercept_only=intercept_curve, slope=slope_curve)


def evaluate_other_state(pq_dir, state, sample_frac, design, models, year_mean):
    """Test-only: score NC-trained models on another state's loans from lenders seen in NC.

    Reported twice: all such loans, and only lender-years with >= MIN_LOANS originations in
    that state (the same volume rule the NC sample uses).
    """
    loans = clean_originations(load_applications(pq_dir, state, originated_only=True))
    qualifying = qualifying_lender_years(loans)
    if sample_frac < 1.0:
        loans = pd.concat([year_loans.sample(frac=sample_frac, random_state=2)
                           for _, year_loans in loans.groupby("activity_year")])
    high_volume = pd.Series([key in qualifying for key in zip(loans.lei, loans.activity_year)], index=loans.index)
    rows = build_features(loans)
    seen = rows[rows.lei.isin(set(design.lenders))]
    samples = {"all_loans_from_nc_lenders": seen,
               "lender_years_with_min_loans_in_state": seen[high_volume.loc[seen.index]]}
    result = dict(rows_all=len(rows))
    for label, sample in samples.items():
        X = design.transform(sample)
        result[label] = dict(n=len(sample), lenders=sample.lei.nunique())
        for name, model in models.items():
            result[label][name] = score_predictions(sample.spread, model.predict(X), year_mean(sample))
    return result


class Eligibility:
    """Which lenders may be recommended to an applicant.

    A lender must be active in 2024 and have >= MIN_BUCKET_LOANS training loans in the
    applicant's bucket (loan-amount quintile x government/conventional), so the model never
    recommends a lender for a product it barely sells.
    """

    def __init__(self, design, train, lenders_active_2024):
        self.amount_edges = np.quantile(train.log_loan_amount, [0.2, 0.4, 0.6, 0.8])
        lender_idx = pd.Index(design.lenders).get_indexer(train.lei)
        buckets = self.bucket(train.log_loan_amount.values, train.is_gov.values)
        self.loans = np.zeros((len(design.lenders), 10))        # lender x bucket loan counts
        np.add.at(self.loans, (lender_idx, buckets), 1)
        self.active = np.array([lei in lenders_active_2024 for lei in design.lenders])

    def bucket(self, log_amount, is_gov):
        return np.digitize(log_amount, self.amount_edges) * 2 + np.asarray(is_gov).astype(int)

    def mask(self, log_amount, is_gov):
        """Boolean table: rows = applicants, columns = lenders."""
        enough = self.loans[:, self.bucket(np.atleast_1d(log_amount), np.atleast_1d(is_gov))] >= MIN_BUCKET_LOANS
        return enough.T & self.active[None, :]


def compare_best_lenders(test, design, models, eligibility, n_applicants=30_000):
    """For sampled test applicants, find each model's cheapest eligible lender and compare."""
    applicants = test.sample(n_applicants, random_state=3)
    eligible = eligibility.mask(applicants.log_loan_amount.values, applicants.is_gov.values)
    z = design.z(applicants)
    slope_scores = models["slope"].lender_scores(z)
    best = {name: np.where(eligible, model.lender_scores(z), np.inf).argmin(axis=1)
            for name, model in models.items()}

    def summarize(winners):
        wins = np.bincount(winners, minlength=len(design.lenders))
        return dict(distinct_best_lenders=int((wins > 0).sum()), modal_lender_share=wins.max() / len(winners))

    row = np.arange(n_applicants)
    # Model-implied saving: slope-model price of the intercept model's pick minus its own pick.
    saving = slope_scores[row, best["intercept_only"]] - slope_scores[row, best["slope"]]
    return dict(n_applicants=n_applicants,
                intercept_only=summarize(best["intercept_only"]), slope=summarize(best["slope"]),
                share_best_differs=float((best["intercept_only"] != best["slope"]).mean()),
                savings_vs_intercept_only_best=dict(mean=saving.mean(), median=np.median(saving),
                                                    p90=np.quantile(saving, 0.9)),
                mean_eligible_lenders_per_applicant=eligible.sum(axis=1).mean())


def profile_inputs(design, eligibility, profile):
    """Standardized slope features and eligible-lender mask for one example applicant."""
    raw = pd.DataFrame([dict(log_loan_amount=np.log(profile["amount"]), ltv=profile["ltv"], dti=profile["dti"],
                             log_income=np.log(profile["income"]), is_gov=profile["gov"])])
    return design.z(raw)[0], eligibility.mask(np.log(profile["amount"]), profile["gov"])[0]


def rank_for_profiles(design, models, eligibility):
    """Five cheapest lenders for each example applicant, priced relative to the median eligible lender."""
    out, top3 = {}, {}
    for name, profile in PROFILES.items():
        z, eligible = profile_inputs(design, eligibility, profile)
        scores = np.where(eligible, models["slope"].lender_scores(z)[0], np.inf)
        flat = models["intercept_only"].lender_intercepts
        order = np.argsort(scores)[:5]
        top3[name] = set(order[:3])
        out[name] = dict(eligible=int(eligible.sum()), top5=[
            dict(lei=str(design.lenders[i]), vs_median_lender=scores[i] - np.median(scores[eligible]),
                 intercept_only_vs_median=flat[i] - np.median(flat[eligible])) for i in order])
    names = list(PROFILES)
    overlap = {f"{a}|{b}": len(top3[a] & top3[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    return out, overlap


def split_half_reliability(design, X_train, y_train, eligibility, fitters):
    """Refit on two disjoint random halves of training; do the lender scores agree?

    `fitters` maps a model name to a function gram -> PriceModel.
    """
    order = np.random.RandomState(5).permutation(X_train.shape[0])
    halves = [order[: len(order) // 2], order[len(order) // 2:]]
    grams = [Gram.from_data(X_train[h], y_train[h]) for h in halves]
    out = {}
    for name, fitter in fitters.items():
        fits = [fitter(g) for g in grams]
        out[name] = {}
        for pname, profile in PROFILES.items():
            z, eligible = profile_inputs(design, eligibility, profile)
            s1, s2 = (m.lender_scores(z)[0] for m in fits)
            top = lambda s: set(np.argsort(np.where(eligible, s, np.inf))[:3])
            out[name][pname] = dict(spearman_halves=spearmanr(s1[eligible], s2[eligible]).correlation,
                                    top3_overlap=len(top(s1) & top(s2)))
    return out


def ols_coefficient_table(design, X_train, train, gram, model):
    """Readable effects from the OLS model with lender intercepts, with lender-clustered 95% intervals."""
    use = model_columns(design)
    cov = clustered_covariance(X_train, train.spread, model, gram, use, X_train[:, design.intercept_cols])
    names = list(design.pooled.get_feature_names_out())
    sd = dict(zip(NUMERIC, design.pooled.named_transformers_["num"].scale_))
    col = lambda name: names.index(name)

    def effect(label, contrast):
        """Estimate and interval for a weighted sum of coefficients."""
        vec = np.zeros(len(use))
        for name, weight in contrast.items():
            vec[col(name)] = weight
        est, se = vec @ model.weights[use], float(np.sqrt(vec @ cov @ vec))
        return dict(effect=label, estimate=est, std_error=se, ci_low=est - 1.96 * se, ci_high=est + 1.96 * se)

    rows = [
        effect("Loan amount doubles", {"num__log_loan_amount": np.log(2) / sd["log_loan_amount"]}),
        effect("Income doubles", {"num__log_income": np.log(2) / sd["log_income"]}),
        effect("LTV +10 points", {"num__ltv": 10 / sd["ltv"]}),
        effect("DTI +10 points", {"num__dti": 10 / sd["dti"]}),
        effect("Tract minority share +10 points", {"num__tract_minority": 10 / sd["tract_minority"]}),
        effect("Tract income (% of area median) +10 points", {"num__tract_income_pct": 10 / sd["tract_income_pct"]}),
    ]
    for code, label in [("2", "FHA"), ("3", "VA"), ("4", "USDA")]:
        rows.append(effect(f"{label} vs conventional", {f"cat__loan_type_{code}": 1, "cat__loan_type_1": -1}))
    for code, label in [("2", "Second home"), ("3", "Investment property")]:
        rows.append(effect(f"{label} vs primary residence", {f"cat__occupancy_type_{code}": 1, "cat__occupancy_type_1": -1}))
    return rows


def residual_checks(test, pred):
    """Mean test residual (actual - predicted) by group; values near zero mean the linear form fits."""
    resid = test.spread.values - pred
    by = lambda keys: {str(k): float(v) for k, v in pd.Series(resid).groupby(np.asarray(keys)).mean().items()}
    decile = pd.qcut(pred, 10, labels=False)
    return dict(by_year=by(test.activity_year), by_loan_type=by(test.loan_type), by_prediction_decile=by(decile))


def lender_table(design, train, models, lenders_active_2024):
    table = pd.DataFrame({
        "lei": design.lenders,
        "n_train": train.groupby("lei").size().reindex(design.lenders).values,
        "active_2024": [lei in lenders_active_2024 for lei in design.lenders],
        "a_intercept_only": models["intercept_only"].lender_intercepts,
        "a_slope": models["slope"].lender_intercepts,
    })
    for k, feature in enumerate(SLOPE_FEATURES):
        table["b_" + feature] = models["slope"].lender_slopes[:, k]
    return table


# ----------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pq", required=True, help="folder holding <STATE>/state_<STATE>_<YEAR>.parquet")
    parser.add_argument("--out", default=".", help="folder for slopes_results.json and lender_params.csv")
    args = parser.parse_args()
    start = time.time()
    log = lambda *msg: print(f"[{time.time() - start:5.0f}s]", *msg, flush=True)

    # 1. Sample and split
    nc, lenders_active_2024 = build_nc_sample(args.pq)
    train, test = nc[nc.split == "train"], nc[nc.split == "test"]
    year_means = train.groupby("activity_year").spread.mean()
    year_mean = lambda df: df.activity_year.map(year_means).fillna(train.spread.mean()).values
    log(f"NC sample: {len(nc):,} loans, {nc.lei.nunique()} lenders")

    # 2. Tune shrinkage
    a_scale, b_scale, tuning = tune_shrinkage(train)
    log(f"chosen scales: intercept {a_scale}, slope {b_scale}")

    # 3. Fit both models on all training rows; test on NC, then SC and TX
    design = LenderDesign(train)
    X_train, X_test = design.transform(train), design.transform(test)
    gram = Gram.from_data(X_train, train.spread)
    models = {"intercept_only": fit_ridge(design, gram, a_scale),
              "slope": fit_ridge(design, gram, a_scale, b_scale)}
    # 6a. Plain OLS versions of the same models, as a baseline
    models.update({"ols_pooled": fit_ols(design, gram, lender_intercepts=False),
                   "ols_intercept_only": fit_ols(design, gram),
                   "ols_slope": fit_ols(design, gram, lender_slopes=True)})
    preds = {name: m.predict(X_test) for name, m in models.items()}
    nc_test = {name: score_predictions(test.spread, p, year_mean(test)) for name, p in preds.items()}
    gain = (test.spread.values - preds["intercept_only"]) ** 2 - (test.spread.values - preds["slope"]) ** 2
    nc_test["paired_mse_gain"] = dict(mean=gain.mean(), se=gain.std() / np.sqrt(len(gain)))
    log("NC test RMSE:", {k: round(v["rmse"], 4) for k, v in nc_test.items() if "rmse" in v})

    ood = {state: evaluate_other_state(args.pq, state, frac, design, models, year_mean)
           for state, frac in OOD_STATES.items()}
    log("SC/TX done")

    # 4. Does the best lender change with the applicant?
    eligibility = Eligibility(design, train, lenders_active_2024)
    ranking = compare_best_lenders(test, design, models, eligibility)
    ols_pair = {"intercept_only": models["ols_intercept_only"], "slope": models["ols_slope"]}
    ranking_ols = compare_best_lenders(test, design, ols_pair, eligibility)
    profiles, profile_overlap = rank_for_profiles(design, models, eligibility)

    # 5. Stability of lender scores
    fitters = {"intercept_only": lambda g: fit_ridge(design, g, a_scale),
               "slope": lambda g: fit_ridge(design, g, a_scale, b_scale),
               "ols_intercept_only": lambda g: fit_ols(design, g),
               "ols_slope": lambda g: fit_ols(design, g, lender_slopes=True)}
    reliability = split_half_reliability(design, X_train, train.spread.values, eligibility, fitters)
    log("ranking and reliability done")

    # 6b. OLS coefficient table and residual checks
    coefficients = ols_coefficient_table(design, X_train, train, gram, models["ols_intercept_only"])
    residuals = residual_checks(test, preds["ols_intercept_only"])
    log("OLS table done")

    results = dict(
        sample=dict(rows=len(nc), lenders=nc.lei.nunique(), train=len(train), test=len(test),
                    active_2024_lenders=len(lenders_active_2024)),
        tuning={k: {str(s): v for s, v in curve.items()} for k, curve in tuning.items()},
        chosen=dict(intercept_scale=a_scale, slope_scale=b_scale, alpha=ALPHA, slope_features=SLOPE_FEATURES),
        NC_test=nc_test, OOD=ood,
        spread_of_lender_effects=dict(
            sd_a_intercept_only=models["intercept_only"].lender_intercepts.std(),
            sd_a_slope=models["slope"].lender_intercepts.std(),
            sd_b=dict(zip(SLOPE_FEATURES, models["slope"].lender_slopes.std(axis=0)))),
        ranking=ranking, ranking_ols=ranking_ols, ols_coefficients=coefficients, ols_residual_means=residuals,
        profiles=profiles, profile_top3_overlap=profile_overlap,
        split_half_reliability=reliability)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "slopes_results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    pd.DataFrame(coefficients).to_csv(os.path.join(args.out, "ols_coefficients.csv"), index=False)
    lender_table(design, train, models, lenders_active_2024).to_csv(
        os.path.join(args.out, "lender_params.csv"), index=False)
    log("saved to", args.out)


if __name__ == "__main__":
    main()
