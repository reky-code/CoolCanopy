import numpy as np
import pandas as pd

from heat.agent.constraints import PlanConstraints
from heat.landsat import dn_to_celsius, qa_masks
from heat.sim.optimize import optimize


def toy():
    # 4 hexes in a row; hex 0 is hottest with most vulnerable people.
    hexes = pd.DataFrame({"hex": [0, 1, 2, 3], "residents": [1000, 800, 500, 300],
                          "vulnerable": [600, 200, 100, 50]})
    sites = pd.DataFrame({
        "site_id": ["a", "b", "c", "d", "elm"],
        "segment_name": ["A St", "B St", "C St", "D St", "Elm St"],
        "hex": [0, 1, 2, 3, 0], "district": ["A", "A", "C", "C", "A"], "capacity": [3, 3, 3, 3, 3],
        "overhead_wires": [False, False, False, False, True],
        "dist_bus_stops_m": [500, 50, 500, 500, 500],
    })
    rows = []
    for sid, h in zip(sites.site_id, sites.hex):
        for t in range(4):
            d = {0: -0.30, 1: -0.08}.get(abs(t - h))
            if d is not None:
                rows.append({"site_id": sid, "target_hex": t, "d_temp_c": d,
                             "d_temp_lo": d * 0.7, "d_temp_hi": d * 1.3})
    return sites, pd.DataFrame(rows), hexes


def test_greedy_prefers_vulnerable_hot_hex_and_logs_exclusion():
    sites, eff, hexes = toy()
    r = optimize(sites, eff, hexes, PlanConstraints(n_trees=4, exclude=["overhead_wires"]),
                 default_cost=1000)
    assert r.picks.iloc[0]["site_id"] == "a"
    assert len(r.picks) == 4
    elm = r.decision_log.set_index("site_id").loc["elm"]
    assert elm["status"] == "excluded" and elm["reason"] == "excluded:overhead_wires"
    lo, hi = r.summary["mean_cooling_range_c"]
    assert lo < r.summary["mean_cooling_benefiting_hexes_c"] < hi


def test_diminishing_returns_gains_never_increase():
    sites, eff, hexes = toy()
    r = optimize(sites, eff, hexes, PlanConstraints(n_trees=10), default_cost=1000)
    g = r.picks["gain"].to_numpy()
    assert np.all(np.diff(g) <= 1e-9)


def test_nested_greedy_prefix():
    sites, eff, hexes = toy()
    small = optimize(sites, eff, hexes, PlanConstraints(n_trees=3), 1000).picks.site_id.tolist()
    big = optimize(sites, eff, hexes, PlanConstraints(n_trees=8), 1000).picks.site_id.tolist()
    assert big[:3] == small


def test_hard_quota_and_budget():
    sites, eff, hexes = toy()
    c = PlanConstraints(budget_usd=6000, n_trees=6,
                        district_prefs=[{"district": "C", "strength": "hard", "min_share": 0.5}])
    r = optimize(sites, eff, hexes, c, default_cost=1000)
    assert r.summary["trees_by_district"].get("C", 0) >= 3
    assert r.summary["cost_usd"] <= 6000


def test_soft_attractor_boosts_bus_stop_site():
    sites, eff, hexes = toy()
    base = optimize(sites, eff, hexes, PlanConstraints(n_trees=12), 1000)
    bus = optimize(sites, eff, hexes, PlanConstraints(n_trees=12, attractors=[
        {"type": "bus_stops", "radius_m": 100, "weight": 3}]), 1000)
    rank = lambda r: r.decision_log.set_index("site_id").loc["b", "first_pick_rank"]
    assert rank(bus) < rank(base)


def test_simulate_sequence_matches_optimizer_and_beats_baselines():
    from heat.sim.optimize import baseline_order, simulate_sequence
    sites, eff, hexes = toy()
    sites["lst_anom_c"] = [1.0, 0.5, 2.0, 0.1, 1.0]
    opt = optimize(sites, eff, hexes, PlanConstraints(n_trees=6), 1000)
    replay = simulate_sequence(sites, eff, hexes, opt.picks.site_id.tolist())
    assert abs(replay.summary["degree_vulnerable_c"] - opt.summary["degree_vulnerable_c"]) < 1e-6
    for strat in ("random", "even", "hottest_first"):
        order = baseline_order(sites, 6, strat)
        assert len(order) == 6
        base = simulate_sequence(sites, eff, hexes, order)
        assert base.summary["degree_vulnerable_c"] <= opt.summary["degree_vulnerable_c"] + 1e-9


def test_qa_and_scaling():
    qa = np.array([0b0100_0000, 0b0000_1000, 0b1100_0000, 0b0001_0000], dtype=np.uint16)
    usable, water = qa_masks(qa)
    assert usable.tolist() == [True, False, True, False]
    assert water.tolist() == [False, False, True, False]
    # DN 44000 -> 44000*0.00341802+149 = 299.39 K = 26.24 C
    assert abs(dn_to_celsius(np.array([44000]), 0.00341802, 149.0)[0] - 26.24) < 0.01

