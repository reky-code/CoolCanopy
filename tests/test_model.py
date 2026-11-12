import h3
import numpy as np
import pandas as pd

from heat.model import FEATURES, planting_effects, train_spatial_cv


def synthetic_city(seed=0):
    rng = np.random.default_rng(seed)
    center = h3.latlng_to_cell(29.76, -95.37, 10)
    cells = list(h3.grid_disk(center, 18))
    df = pd.DataFrame({"cell": cells})
    df["h3_10"] = [h3.str_to_int(c) for c in cells]
    ll = np.array([h3.cell_to_latlng(c) for c in cells])
    df["lat"], df["lon"] = ll[:, 0], ll[:, 1]
    df["canopy_pct_2021"] = np.clip(rng.normal(20, 12, len(df)), 0, 90)
    df["imperv_pct_2021"] = np.clip(rng.normal(50, 20, len(df)), 0, 100)
    idx = {c: i for i, c in enumerate(cells)}
    for col, base in (("canopy", "canopy_pct_2021"), ("imperv", "imperv_pct_2021")):
        for k in (1, 2):
            df[f"{col}_k{k}"] = [np.mean([df[base].iat[idx[n]] for n in h3.grid_disk(c, k)
                                          if n in idx]) for c in cells]
    df["water_frac"] = 0.0
    df["lst_anom_c"] = (0.10 * df.imperv_pct_2021 - 0.15 * df.canopy_pct_2021
                        - 0.05 * df.canopy_k2 + rng.normal(0, 0.5, len(df)))
    df["parent_r6"] = [h3.cell_to_parent(c, 8) for c in cells]   # small toy area
    return df


def test_train_and_effects_are_sane():
    df = synthetic_city()
    res = train_spatial_cv(df, n_folds=4, num_rounds=150, random_cv_compare=False)
    assert res.metrics["spatial_cv_r2"] > 0.6
    sites = df.sample(30, random_state=1)
    pairs = []
    for s in sites.cell:
        for t in h3.grid_disk(s, 2):
            pairs.append({"site_hex": h3.str_to_int(s), "target_hex": h3.str_to_int(t),
                          "k": h3.grid_distance(s, t)})
    eff = planting_effects(df[["h3_10", *FEATURES]], pd.DataFrame(pairs),
                           [res.full_model, *res.fold_models])
    assert len(eff) > 0
    assert (eff.d_temp_c <= 0).all()
    own = eff[eff.k == 0].d_temp_c.mean()
    ring2 = eff[eff.k == 2].d_temp_c.mean()
    # one tree ~0.33 pp canopy at its cell; true slope 0.15 C/pp -> about -0.05 C at k=0
    assert -0.15 < own < -0.005
    assert own < ring2 <= 0          # cooling fades with distance
    assert (eff.d_temp_lo <= eff.d_temp_hi + 1e-12).all()
