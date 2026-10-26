"""Data loading and explicitly-labelled synthetic fallback for FinSight AI."""
import numpy as np
import pandas as pd
from pathlib import Path


def load_local_arff(path=None):
    """Load the official 3year.arff file downloaded from UCI, if present."""
    from scipy.io import arff
    data_path = Path(path) if path else Path(__file__).resolve().parent / "data" / "3year.arff"
    if not data_path.exists():
        raise FileNotFoundError(f"Local UCI file not found: {data_path}")
    records, _ = arff.loadarff(str(data_path))
    frame = pd.DataFrame(records)
    # ARFF string fields may be bytes; normalize them before numeric conversion.
    for col in frame.columns:
        if frame[col].dtype == object:
            frame[col] = frame[col].map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)
    target_col = next((c for c in frame.columns if str(c).lower() in {"class", "bankrupt", "target"}), frame.columns[-1])
    target = pd.to_numeric(frame.pop(target_col), errors="coerce")
    X = frame.apply(pd.to_numeric, errors="coerce")
    valid = target.isin([0, 1])
    X = X.loc[valid].reset_index(drop=True)
    target = target.loc[valid].astype(int).reset_index(drop=True)
    if set(target.unique()) != {0, 1}:
        raise ValueError(f"Expected bankruptcy labels 0 and 1; got {sorted(target.unique())}")
    label = "REAL DATA — UCI Polish Companies Bankruptcy, 3year.arff cohort; class 1 = bankrupt, class 0 = non-bankrupt"
    return X, target.rename("bankruptcy"), label


def load_uci_bankruptcy(cohort_year: int = 3):
    """Load the UCI Polish bankruptcy data and keep one forecast cohort.

    UCI's combined endpoint can expose multiple yearly cohorts. Their target
    horizons differ, so this project uses only the 3rd-year cohort to avoid
    mixing outcome horizons in one evaluation. Class 1 means bankrupt.
    """
    # Prefer the official local ARFF file; this also works when UCI API access is blocked.
    local_file = Path(__file__).resolve().parent / "data" / "3year.arff"
    if local_file.exists():
        return load_local_arff(local_file)

    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=365)
    X = dataset.data.features.copy()
    target_frame = dataset.data.targets.copy()
    if target_frame.shape[1] != 1:
        raise ValueError(f"Expected one target column, got {target_frame.shape[1]}")
    target = pd.to_numeric(target_frame.iloc[:, 0], errors="coerce")

    # Keep the same forecasting cohort for every row. The UCI combined table
    # contains a 'year' field; if absent, assume the repository returned the
    # cohort-specific table directly.
    if "year" in X.columns:
        cohort = pd.to_numeric(X["year"], errors="coerce")
        mask = cohort.eq(cohort_year) & target.notna()
        X = X.loc[mask].drop(columns=["year"]).reset_index(drop=True)
        target = target.loc[mask].reset_index(drop=True)
    else:
        mask = target.notna()
        X = X.loc[mask].reset_index(drop=True)
        target = target.loc[mask].reset_index(drop=True)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, ~X.columns.duplicated()]

    unique_values = set(target.dropna().unique().tolist())
    if not unique_values.issubset({0, 1}) or unique_values != {0, 1}:
        raise ValueError(f"Expected target labels 0 and 1, got {sorted(unique_values)}")
    y = target.astype(int).rename("bankruptcy")
    label = (
        "REAL DATA — UCI Polish Companies Bankruptcy, 3rd-year cohort; "
        "class 1 = bankrupt, class 0 = non-bankrupt"
    )
    return X, y, label


def make_synthetic_demo(n=1600, seed=42):
    """Generate a synthetic fallback for software testing only."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "current_ratio": rng.normal(1.6, .55, n).clip(.1, 5),
        "debt_to_assets": rng.beta(2.2, 3, n).clip(.01, .99),
        "profit_margin": rng.normal(.08, .12, n).clip(-.6, .6),
        "asset_turnover": rng.normal(.9, .35, n).clip(.05, 3),
        "revenue_growth": rng.normal(.06, .18, n).clip(-.8, 1.2),
        "cash_to_assets": rng.beta(2, 5, n).clip(.001, .9),
        "interest_coverage": rng.lognormal(1.1, .8, n).clip(.01, 20),
        "volatility_proxy": rng.beta(2, 4, n).clip(.001, .99),
    })
    score = (2.1 * X.debt_to_assets - X.current_ratio - 2.5 * X.profit_margin
             - .7 * X.asset_turnover - X.revenue_growth - 1.4 * X.cash_to_assets
             - .12 * X.interest_coverage + 1.1 * X.volatility_proxy
             + rng.normal(0, .45, n))
    y = (score > np.quantile(score, .82)).astype(int)
    return X, pd.Series(y, name="bankruptcy"), "SYNTHETIC DEMO DATA — NOT FOR RESEARCH CLAIMS"
