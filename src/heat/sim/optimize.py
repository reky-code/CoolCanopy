"""Lazy-greedy (CELF) tree-planting optimizer with a per-site decision log.

Inputs (pandas DataFrames, produced by the gold layer):
  sites    site_id, hex, district, capacity, [cost], [segment_name],
           overhead_wires, transmission, narrow_sidewalk (bool), [dist_bus_stops_m, ...]
  effects  site_id, target_hex, d_temp_c [, d_temp_lo, d_temp_hi]
           Temperature change at target_hex from ONE young tree at site (negative = cooler),
           precomputed once by re-scoring the heat model with the added crown.
  hexes    hex, residents, vulnerable

Diminishing returns: the marginal effect at a hex shrinks as that hex is already cooled,
  factor = exp(-cooled_so_far / saturation_c). Gains only shrink as trees are added, so
  lazy evaluation is valid and the greedy order is nested (first k picks of N = best k),
  which is what makes the app's "plant N trees" slider a lookup.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from heat.agent.constraints import PlanConstraints, is_district_code

BENEFIT_THRESHOLD_C = 0.05  # a resident "benefits" if their hex cools by at least this


@dataclass
class PlanResult:
    picks: pd.DataFrame            # order, site_id, hex, district, gain, cost, cum_cost
    hex_delta: pd.DataFrame        # hex, d_temp_c [, d_temp_lo, d_temp_hi]
    decision_log: pd.DataFrame     # site_id, status, reason, trees, rank, gain_ratio
    summary: dict[str, Any] = field(default_factory=dict)
    # n -> (hex_delta, summary) snapshots of the first n trees (slider frames)
    frames: dict[int, tuple[pd.DataFrame, dict[str, Any]]] = field(default_factory=dict)


def in_area(sites: pd.DataFrame, area: str) -> np.ndarray:
    """Sites inside a council district ("J") or a Super Neighborhood ("GULFTON", case-insensitive)."""
    if is_district_code(area):
        return (sites["district"].astype(str).str.upper() == str(area).upper()).to_numpy()
    if "super_nbhd" not in sites:
        return np.zeros(len(sites), bool)
    return (sites["super_nbhd"].fillna("").str.upper() == str(area).upper()).to_numpy()


def _effect_vectors(effects: pd.DataFrame, site_ids: pd.Series, hex_ids: pd.Index):
    """Per-site (target positions, cooling magnitude, lo, hi) as views into flat arrays.

    One vectorized pass: map ids to positions with Index.get_indexer, sort by site, and cut at
    np.searchsorted boundaries. (A per-site groupby + Series.map(dict) rebuilt a 105k-entry
    Series for each of 81k sites and never finished at Houston scale.)"""
    hex_ids = pd.Index(hex_ids)
    site_index = pd.Index(site_ids)
    t_pos = hex_ids.get_indexer(effects["target_hex"])
    s_pos = site_index.get_indexer(effects["site_id"])
    keep = (t_pos >= 0) & (s_pos >= 0)
    has_ci = {"d_temp_lo", "d_temp_hi"} <= set(effects.columns)
    s_pos, t_pos = s_pos[keep], t_pos[keep]
    order = np.argsort(s_pos, kind="stable")
    s_pos, t_pos = s_pos[order], t_pos[order].astype(np.int64)
    m = -effects["d_temp_c"].to_numpy(float)[keep][order]          # cooling magnitude, >= 0
    l = -effects["d_temp_lo"].to_numpy(float)[keep][order] if has_ci else np.zeros_like(m)
    h = -effects["d_temp_hi"].to_numpy(float)[keep][order] if has_ci else np.zeros_like(m)
    bounds = np.searchsorted(s_pos, np.arange(len(site_index) + 1))
    sl = [slice(bounds[i], bounds[i + 1]) for i in range(len(site_index))]
    return ([t_pos[s] for s in sl], [m[s] for s in sl], [l[s] for s in sl],
            [h[s] for s in sl], has_ci)


def optimize(sites: pd.DataFrame, effects: pd.DataFrame, hexes: pd.DataFrame,
             c: PlanConstraints, default_cost: float, saturation_c: float = 1.5) -> PlanResult:
    sites = sites.reset_index(drop=True).copy()
    if "cost" not in sites:
        sites["cost"] = np.nan
    sites["cost"] = sites["cost"].fillna(c.effective_cost(default_cost))
    n_target = c.resolved_n_trees(default_cost)
    budget = c.budget_usd if c.budget_usd else math.inf

    # ---- exclusions -> decision log ---------------------------------------------------
    reason = pd.Series("", index=sites.index, dtype=object)
    for ex in c.exclude:
        if ex in sites:
            hit = sites[ex].fillna(False).astype(bool) & (reason == "")
            reason[hit] = f"excluded:{ex}"
    eligible = reason == ""

    # ---- hex weights by objective -----------------------------------------------------
    hexes = hexes.set_index("hex")
    if c.objective == "vulnerability":
        w = hexes["vulnerable"]
    elif c.objective == "cooling":
        w = hexes["residents"]
    else:
        w = 0.5 * hexes["residents"] + 0.5 * hexes["vulnerable"]
    hex_index = {h: i for i, h in enumerate(hexes.index)}
    w_arr = w.to_numpy(dtype=float)

    # ---- site multipliers: soft district prefs + attractors ---------------------------
    mult = np.ones(len(sites))
    for p in c.district_prefs:
        if p.strength == "soft":
            mult[in_area(sites, p.district)] *= p.weight
    for a in c.attractors:
        col = f"dist_{a.type}_m"
        if col in sites:
            mult[(sites[col] <= a.radius_m).fillna(False).to_numpy()] *= a.weight

    # ---- per-site effect vectors ------------------------------------------------------
    tgt, mag, lo, hi, has_ci = _effect_vectors(effects, sites["site_id"], hexes.index)

    cooled = np.zeros(len(hexes))     # cumulative cooling magnitude per hex (point est.)
    cooled_lo = np.zeros(len(hexes))
    cooled_hi = np.zeros(len(hexes))
    planted = np.zeros(len(sites), int)
    capacity = sites["capacity"].fillna(1).astype(int).to_numpy()
    cost = sites["cost"].to_numpy(float)

    def gain(i: int) -> float:
        t = tgt[i]
        if t.size == 0:
            return 0.0
        f = np.exp(-cooled[t] / saturation_c)
        return float(mult[i] * np.sum(mag[i] * f * w_arr[t]))

    picks: list[dict[str, Any]] = []
    spent = 0.0
    first_rank = np.full(len(sites), -1)
    last_gain = np.zeros(len(sites))

    def run(pool: np.ndarray, n_max: int) -> None:
        nonlocal spent
        heap = [(-gain(i), i) for i in np.flatnonzero(pool)]
        heapq.heapify(heap)
        n_done = 0
        while heap and n_done < n_max and len(picks) < n_target:
            neg_ub, i = heapq.heappop(heap)
            if planted[i] >= capacity[i]:
                continue
            if spent + cost[i] > budget:
                last_gain[i] = -neg_ub
                continue                                    # unaffordable; drop
            g = gain(i)                                     # re-evaluate lazily
            last_gain[i] = g
            if heap and g < -heap[0][0] - 1e-12:
                heapq.heappush(heap, (-g, i))
                continue
            if g <= 0:
                break
            # accept one tree at site i
            t = tgt[i]
            f = np.exp(-cooled[t] / saturation_c)
            cooled[t] += mag[i] * f
            if has_ci:
                cooled_lo[t] += lo[i] * f
                cooled_hi[t] += hi[i] * f
            planted[i] += 1
            spent += cost[i]
            n_done += 1
            if first_rank[i] < 0:
                first_rank[i] = len(picks) + 1
            picks.append({"order": len(picks) + 1, "site_id": sites.at[i, "site_id"],
                          "hex": sites.at[i, "hex"], "district": sites.at[i, "district"],
                          "gain": g, "cost": cost[i], "cum_cost": spent})
            if planted[i] < capacity[i]:
                heapq.heappush(heap, (-gain(i), i))

    # Phase 1: hard district quotas. Phase 2: everything else.
    for p in c.district_prefs:
        if p.strength == "hard" and p.min_share:
            pool = eligible.to_numpy() & in_area(sites, p.district)
            run(pool, math.ceil(p.min_share * n_target))
    run(eligible.to_numpy(), n_target)

    picks_df = pd.DataFrame(picks, columns=["order", "site_id", "hex", "district", "gain",
                                            "cost", "cum_cost"])
    cutoff = picks_df["gain"].min() if len(picks_df) else 0.0

    # ---- decision log -----------------------------------------------------------------
    status = np.where(planted > 0, "selected", np.where(reason != "", "excluded", "not_selected"))
    final_reason = reason.copy()
    for i in np.flatnonzero((planted == 0) & (reason == "").to_numpy()):
        if spent + cost[i] > budget and len(picks) < n_target:
            final_reason[i] = "budget"
        elif tgt[i].size == 0:
            final_reason[i] = "no_modeled_benefit"
        else:
            final_reason[i] = "outranked"
    final_reason[planted > 0] = np.where(planted[planted > 0] >= capacity[planted > 0],
                                         "selected:capacity_full", "selected")
    log = pd.DataFrame({
        "site_id": sites["site_id"], "status": status, "reason": final_reason,
        "trees": planted, "first_pick_rank": first_rank,
        "marginal_gain": [last_gain[i] if last_gain[i] else gain(i) for i in range(len(sites))],
    })
    log["gain_vs_cutoff"] = np.where(cutoff > 0, log["marginal_gain"] / cutoff, np.nan)
    if "segment_name" in sites:
        log.insert(1, "segment_name", sites["segment_name"])

    # ---- hex deltas + summary ---------------------------------------------------------
    hd = pd.DataFrame({"hex": hexes.index, "d_temp_c": -cooled})
    if has_ci:
        hd["d_temp_lo"] = -cooled_lo
        hd["d_temp_hi"] = -cooled_hi
    hd = hd[hd["d_temp_c"] < 0].reset_index(drop=True)

    summary = summarize(hd, hexes.reset_index(), picks_df, n_target, spent)
    return PlanResult(picks_df, hd, log, summary)


def simulate_sequence(sites: pd.DataFrame, effects: pd.DataFrame, hexes: pd.DataFrame,
                      order: list[Any], saturation_c: float = 1.5,
                      cost_per_tree: float = 0.0,
                      checkpoints: list[int] | None = None) -> PlanResult:
    """Score a given planting order (site_ids, repeats = several trees at one site) with the
    same effect table and diminishing-returns rule the optimizer uses. Baseline strategies
    (random, even, hottest-first) go through here so the comparison is apples to apples."""
    hx = hexes.set_index("hex")
    site_ids = pd.Index(sites["site_id"].unique())
    tgt, mag, lo, hi, has_ci = _effect_vectors(effects, site_ids.to_series(), hx.index)
    pos = {sid: i for i, sid in enumerate(site_ids)}
    cooled = np.zeros(len(hx)); clo = np.zeros(len(hx)); chi = np.zeros(len(hx))
    site_info = sites.drop_duplicates("site_id").set_index("site_id")
    site_hex = site_info["hex"].to_dict()
    site_district = site_info["district"].to_dict()
    picks = []
    cps = set(checkpoints or [])
    frames: dict[int, tuple[pd.DataFrame, dict[str, Any]]] = {}

    def snapshot(n_done: int) -> tuple[pd.DataFrame, dict[str, Any]]:
        hd = pd.DataFrame({"hex": hx.index, "d_temp_c": -cooled})
        if has_ci:
            hd["d_temp_lo"], hd["d_temp_hi"] = -clo, -chi
        hd = hd[hd["d_temp_c"] < 0].reset_index(drop=True)
        return hd, summarize(hd, hexes, pd.DataFrame(picks), n_done, n_done * cost_per_tree)

    for n, sid in enumerate(order, 1):
        i = pos.get(sid)
        if i is not None and tgt[i].size:
            t = tgt[i]
            f = np.exp(-cooled[t] / saturation_c)
            cooled[t] += mag[i] * f
            if has_ci:
                clo[t] += lo[i] * f
                chi[t] += hi[i] * f
        picks.append({"order": n, "site_id": sid, "hex": site_hex.get(sid),
                      "district": site_district.get(sid), "gain": np.nan,
                      "cost": cost_per_tree, "cum_cost": n * cost_per_tree})
        if n in cps:
            frames[n] = snapshot(n)
    hd, summary = snapshot(len(order))
    return PlanResult(pd.DataFrame(picks), hd, pd.DataFrame(), summary, frames)


def baseline_order(sites: pd.DataFrame, n: int, strategy: str, seed: int = 7) -> list[Any]:
    """Planting orders for the comparison strategies. Each site contributes `capacity` slots."""
    rng = np.random.default_rng(seed)
    slots = sites.loc[sites.index.repeat(sites["capacity"].clip(lower=1).astype(int))]
    if strategy == "random":
        return slots["site_id"].sample(frac=1, random_state=seed).head(n).tolist()
    if strategy == "hottest_first":
        return slots.sort_values("lst_anom_c", ascending=False)["site_id"].head(n).tolist()
    if strategy == "even":   # equal share per district, random within district
        per = {d: g["site_id"].sample(frac=1, random_state=seed).tolist()
               for d, g in slots.groupby(slots["district"].fillna("none"))}
        order, i = [], 0
        keys = sorted(per)
        while len(order) < n and any(per.values()):
            k = keys[i % len(keys)]
            if per[k]:
                order.append(per[k].pop())
            i += 1
        return order
    raise ValueError(strategy)


def summarize(hex_delta: pd.DataFrame, hexes: pd.DataFrame, picks: pd.DataFrame,
              n_target: int, spent: float) -> dict[str, Any]:
    m = hex_delta.merge(hexes, on="hex", how="left")
    ben = m[m["d_temp_c"] <= -BENEFIT_THRESHOLD_C]
    out: dict[str, Any] = {
        "n_trees_requested": int(n_target),
        "n_trees_planned": int(len(picks)),
        "cost_usd": round(float(spent), 2),
        "residents_benefiting": int(ben["residents"].sum()),
        "vulnerable_benefiting": int(ben["vulnerable"].sum()),
        "degree_person_c": round(float((-m["d_temp_c"] * m["residents"]).sum()), 1),
        "degree_vulnerable_c": round(float((-m["d_temp_c"] * m["vulnerable"]).sum()), 1),
        "mean_cooling_benefiting_hexes_c": round(float(-ben["d_temp_c"].mean()), 3)
        if len(ben) else 0.0,
    }
    if {"d_temp_lo", "d_temp_hi"} <= set(m.columns) and len(ben):
        a, b = -ben["d_temp_lo"].mean(), -ben["d_temp_hi"].mean()
        out["mean_cooling_range_c"] = [round(float(min(a, b)), 3), round(float(max(a, b)), 3)]
    if len(picks):
        out["trees_by_district"] = {str(k): int(v) for k, v in
                                    picks["district"].value_counts().sort_index().items()}
    return out
