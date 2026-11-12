"""Heat model: LightGBM with physically-signed monotone constraints and spatial CV, plus
the planting-effect table the optimizer consumes.

Why monotone constraints: the simulator asks "what if canopy goes up here?". An
unconstrained tree model can answer "hotter" in places just because of noise. Constraining
canopy to never increase heat and pavement to never decrease it keeps the
what-if physically sane without forcing a functional form.

Why spatial CV: neighbouring hexes share pixels (thermal is ~100 m native) and ring
features, so a random K-fold leaks. We hold out whole H3 res-6 parents (~36 km2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

FEATURES = ["canopy_pct_2021", "canopy_k1", "canopy_k2",
            "imperv_pct_2021", "imperv_k1", "imperv_k2",
            "water_frac", "lat", "lon"]
MONOTONE = {"canopy_pct_2021": -1, "canopy_k1": -1, "canopy_k2": -1,
            "imperv_pct_2021": 1, "imperv_k1": 1, "imperv_k2": 1,
            "water_frac": -1, "lat": 0, "lon": 0}
TARGET = "lst_anom_c"

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression", "learning_rate": 0.05, "num_leaves": 63,
    "min_data_in_leaf": 50, "feature_fraction": 0.9, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": 7,
    "monotone_constraints_method": "advanced",
}


@dataclass
class TrainResult:
    full_model: Any
    fold_models: list[Any]
    metrics: dict[str, float]
    fold_metrics: pd.DataFrame
    oof_pred: np.ndarray = field(repr=False, default=None)


def _params(features: list[str]) -> dict[str, Any]:
    return {**DEFAULT_PARAMS, "monotone_constraints": [MONOTONE.get(f, 0) for f in features]}


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def train_spatial_cv(df: pd.DataFrame, group_col: str = "parent_r6", n_folds: int = 5,
                     num_rounds: int = 600, features: list[str] = FEATURES,
                     random_cv_compare: bool = True) -> TrainResult:
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold, KFold

    X, y, g = df[features].to_numpy(float), df[TARGET].to_numpy(float), df[group_col].to_numpy()
    params = _params(features)
    oof = np.zeros(len(df))
    folds, rows = [], []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, g)):
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr], feature_name=features), num_rounds)
        oof[te] = m.predict(X[te])
        folds.append(m)
        rows.append({"fold": k, "n_test": len(te), "r2": _r2(y[te], oof[te]),
                     "mae_c": float(np.mean(np.abs(y[te] - oof[te])))})
    metrics = {"spatial_cv_r2": _r2(y, oof), "spatial_cv_mae_c": float(np.mean(np.abs(y - oof)))}
    if random_cv_compare:   # shows how much a naive split would flatter the model
        rp = np.zeros(len(df))
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=7).split(X):
            rp[te] = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_rounds).predict(X[te])
        metrics["random_cv_r2"] = _r2(y, rp)
    full = lgb.train(params, lgb.Dataset(X, y, feature_name=features), num_rounds)
    return TrainResult(full, folds, metrics, pd.DataFrame(rows), oof)


def ols_baseline(df: pd.DataFrame, group_col: str = "parent_r6") -> pd.DataFrame:
    """Interpretable headline: OLS with SEs clustered by res-6 parent (spatial autocorr.)."""
    import statsmodels.api as sm

    cols = ["canopy_pct_2021", "canopy_k2", "imperv_pct_2021", "imperv_k2", "water_frac"]
    X = sm.add_constant(df[cols].astype(float))
    fit = sm.OLS(df[TARGET].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df[group_col])[0]})
    out = pd.DataFrame({"coef": fit.params, "se": fit.bse, "p": fit.pvalues})
    out["per_10pct_c"] = out["coef"] * 10
    out.attrs["r2"] = float(fit.rsquared)
    return out


# ---- planting effects ----------------------------------------------------------------

H3_RES10_AREA_M2 = 15_047.5   # average res-10 cell area


def planting_effects(features: pd.DataFrame, pairs: pd.DataFrame, models: list[Any],
                     crown_m2: float = 50.0, block_trees: int = 20,
                     feature_cols: list[str] = FEATURES) -> pd.DataFrame:
    """Temperature change at each target cell from ONE young tree at each site.

    features: h3_10 + feature_cols (one row per cell)
    pairs:    site_hex, target_hex, k (H3 grid distance 0..2), from SQL h3_kring/h3_distance
    models:   [point_model, *ensemble]; ensemble spread gives d_temp_lo / d_temp_hi

    A block of `block_trees` is simulated and divided back, because a tree model is
    piecewise-constant: one tree's 0.3 pp canopy bump would usually change nothing.
    Ring features are means over 7 (k<=1) and 19 (k<=2) cells, so a canopy bump at the
    site raises a neighbour's ring mean by bump/7 or bump/19.
    """
    bump = block_trees * crown_m2 / H3_RES10_AREA_M2 * 100.0          # canopy pct points
    f = features.set_index("h3_10")
    p = pairs[pairs["target_hex"].isin(f.index)].reset_index(drop=True)
    base = f.loc[p["target_hex"], feature_cols].reset_index(drop=True)
    new = base.copy()
    k = p["k"].to_numpy()
    new["canopy_pct_2021"] = np.minimum(100, base["canopy_pct_2021"] + np.where(k == 0, bump, 0))
    new["canopy_k1"] = np.minimum(100, base["canopy_k1"] + np.where(k <= 1, bump / 7, 0))
    new["canopy_k2"] = np.minimum(100, base["canopy_k2"] + np.where(k <= 2, bump / 19, 0))
    Xb, Xn = base.to_numpy(float), new.to_numpy(float)
    deltas = np.column_stack([m.predict(Xn) - m.predict(Xb) for m in models]) / block_trees
    deltas = np.minimum(deltas, 0.0)     # monotone => <= 0 up to float noise
    out = pd.DataFrame({"site_hex": p["site_hex"], "target_hex": p["target_hex"], "k": k,
                        "d_temp_c": deltas[:, 0]})
    if deltas.shape[1] > 1:
        ens = deltas[:, 1:]
        out["d_temp_lo"] = np.quantile(ens, 0.1, axis=1)   # more cooling
        out["d_temp_hi"] = np.quantile(ens, 0.9, axis=1)   # less cooling
    return out[out[["d_temp_c"] + [c for c in ("d_temp_lo",) if c in out]].min(axis=1) < 0]
