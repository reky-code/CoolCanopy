import pandas as pd
import pytest

from heat import game as g


def board():
    # 5 hexes in a row, all in "Testville". Hex 1 is crowded, hex 4 is empty but hot.
    hexes = pd.DataFrame({"hex": [0, 1, 2, 3, 4], "residents": [300, 1200, 400, 100, 0],
                          "vulnerable": [250, 200, 100, 80, 0]})
    sites = pd.DataFrame({
        "site_id": ["s0", "s1", "s2", "s3", "s4"], "hex": [0, 1, 2, 3, 4],
        "district": ["J"] * 5, "super_nbhd": ["Testville"] * 5,
        "capacity": [3, 3, 3, 3, 3], "imperv_pct_2021": [30, 80, 40, 30, 90],
        "lst_anom_c": [2, 3, 2, 1, 6], "canopy_pct_2021": [10, 5, 10, 30, 0],
        "residents": [300, 1200, 400, 100, 0], "vulnerable": [250, 200, 100, 80, 0],
    })
    rows = []
    for sid, h in zip(sites.site_id, sites.hex):
        for t in range(5):
            d = {0: -0.30, 1: -0.08}.get(abs(t - h))
            if d is not None:
                rows.append({"site_id": sid, "target_hex": t, "d_temp_c": d})
    return g.Board(sites, pd.DataFrame(rows), hexes, "Testville")


R1 = g.Round("t1", "t", "t", "cooling", n_trees=3)
R2 = g.Round("t2", "t", "t", "cooling", budget_usd=3000, cost_open=1000, cost_paved=2500)
R3 = g.Round("t3", "t", "t", "vulnerability", n_trees=3)


def test_unknown_area_raises():
    b = board()
    with pytest.raises(ValueError):
        g.Board(b.sites, b.effects, b.hexes, "Nowhere")


def test_optimal_beats_hot_but_empty_cell():
    b = board()
    best = b.evaluate(b.optimal_order(R1), R1)
    hot = b.evaluate(["s4", "s4", "s4"], R1)
    assert best.trees == 3 and best.score > 2 * hot.score
    assert g.pct_of(best, best) == pytest.approx(100)
    assert b.optimal_order(R1)[0] == "s1"


def test_diminishing_returns_per_tree():
    b = board()
    o = b.evaluate(["s1", "s1", "s1"], R1)
    assert o.per_tree[0] > o.per_tree[1] > o.per_tree[2] > 0
    assert b.marginal(["s1"], "s1", R1) == pytest.approx(o.per_tree[1])


def test_rules_capacity_trees_budget():
    b = board()
    assert not b.can_plant(["s0"] * 3, "s0", g.Round("x", "", "", "cooling", n_trees=9))[0]
    assert not b.can_plant(["s0", "s2", "s3"], "s1", R1)[0]          # out of trees
    assert b.can_plant(["s0"], "s2", R2)[0]                           # 1000 + 1000
    assert not b.can_plant(["s0"], "s1", R2)[0]                       # 1000 + 2500 > 3000
    assert b.optimal_order(R2) and b.evaluate(b.optimal_order(R2), R2).spent <= 3000


def test_equity_round_shifts_optimum():
    b = board()
    assert b.optimal_order(R3)[0] == "s0"     # most vulnerable, not most residents


def test_baselines_respect_limits():
    b = board()
    assert len(b.baseline(R1, "random")) == 3
    assert b.evaluate(b.baseline(R2, "hottest_first"), R2).spent <= 3000


def test_coaching_and_grades():
    b = board()
    best_order = b.optimal_order(R1)
    me = b.evaluate(["s4", "s3", "s3"], R1)
    c = g.coaching(b, ["s4", "s3", "s3"], best_order, me)
    assert c["missed"][0]["site_id"] == "s1"
    assert c["worst"]["site_id"] == "s4"
    assert "almost nobody" in g.why(c["worst"])
    best = b.evaluate(best_order, R1)
    assert g.final_pct(best, best, {"residents"}) == pytest.approx(95)
    assert g.grade(96) == "S" and g.grade(71) == "B" and g.grade(10) == "D"
    names = [n for n, _ in g.achievements(R1, best, best, 100, set())]
    assert "Beat the algorithm" in names and "Gut instinct" in names


def test_done_needs_all_trees_or_budget():
    b = board()
    assert not b.done(["s0", "s2"], R1) and b.done(["s0", "s2", "s3"], R1)
    assert not b.done(["s0"], R2)                      # $2,000 left: one more open-ground tree fits
    assert b.done(["s0", "s2", "s3"], R2)              # $0 left


def test_why_converts_units():
    f = {"first_c": 0.5, "people_reached": 900, "residents": 200, "heat_c": 4, "canopy_pct": 0}
    w = g.why(f, 1.8, "°F")
    assert "cools this block by 0.90 °F" in w and "about 900 people" in w and "+7.2 °F" in w


def test_degrees_per_tree_and_next_tree_preview():
    b = board()
    assert b.sites.set_index("site_id").at["s1", "first_c"] == pytest.approx(0.30)
    o = b.evaluate(["s1", "s1"], R1)
    assert o.per_tree_c[0] == pytest.approx(0.30) and 0 < o.per_tree_c[1] < 0.30   # diminishing
    assert o.peak_c == pytest.approx(0.30 + o.per_tree_c[1])
    assert 0 < o.mean_c <= o.peak_c
    nxt = b.next_tree(["s1", "s1"], "s1")
    assert nxt["own_c"] == pytest.approx(b.evaluate(["s1"] * 3, R1).per_tree_c[2])
    assert nxt["people"] == pytest.approx(300 + 1200 + 400)
