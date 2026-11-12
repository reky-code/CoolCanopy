""""Beat the Algorithm": a planting game on one neighborhood.

The player places trees one at a time on the neighborhood's planting sites. Every tree is scored
with the same effect table and diminishing-returns rule the optimizer uses, so "you reached 71% of
optimal" compares like with like. Three rounds change the rules (and so the optimal map):

  1. Cool the crowd       N trees, every resident counts (objective "cooling")
  2. Tight budget         a dollar budget; paved cells cost more (concrete cutting, tree pits)
  3. Protect the vulnerable  N trees, only vulnerable residents count (objective "vulnerability")

Pure pandas/numpy, no Streamlit: the app keeps a Board in a cache and the player's click order in
session state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from heat.agent.constraints import PlanConstraints
from heat.sim.optimize import (BENEFIT_THRESHOLD_C, _effect_vectors, baseline_order, in_area,
                               optimize)


@dataclass(frozen=True)
class Round:
    key: str
    title: str
    brief: str
    objective: str                 # "cooling" (residents) | "vulnerability" (vulnerable residents)
    n_trees: int | None = None
    budget_usd: float | None = None
    cost_open: float = 1500.0      # $/tree on open ground
    cost_paved: float = 1500.0     # $/tree where the cell is mostly paved
    paved_pct: float = 60.0        # imperv % above which a cell counts as paved


ROUNDS: list[Round] = [
    Round("crowd", "Cool the crowd",
          "You have 40 trees. Every resident who gets cooler counts.", "cooling", n_trees=40),
    Round("budget", "Tight budget",
          "You have $60,000. Open ground costs $1,000 a tree. Paved blocks need concrete cut "
          "and a tree pit: $2,500.", "cooling", budget_usd=60_000, cost_open=1000, cost_paved=2500),
    Round("equity", "Protect the vulnerable",
          "40 trees again, but cooling now counts by social vulnerability (CDC SVI: poverty, age, "
          "disability, language, no car, crowded housing). See FAQ & definitions in the sidebar.", "vulnerability", n_trees=40),
]

GRADES = [(95, "S"), (85, "A"), (70, "B"), (50, "C"), (0, "D")]
INTEL_COST = {"residents": 5, "vulnerability": 5}   # points off the final % for each paid hint


def grade(pct: float) -> str:
    return next(g for lo, g in GRADES if pct >= lo)


@dataclass
class Outcome:
    score: float                   # weighted person-degrees (the objective)
    residents: int                 # residents in cells cooled >= threshold
    vulnerable: int
    cells_cooled: int
    spent: float
    trees: int
    per_tree: list[float] = field(default_factory=list)   # marginal score of each tree, in order
    per_tree_c: list[float] = field(default_factory=list) # °C each tree cooled its own cell, in order
    peak_c: float = 0.0            # biggest drop in any cell (°C)
    mean_c: float = 0.0            # average drop for residents in cooled cells (°C)
    hex_delta: pd.DataFrame | None = None                 # hex, d_temp_c (< 0)
    counts: dict[Any, int] = field(default_factory=dict)  # site_id -> trees


class Board:
    """Planting sites of one area plus everything needed to score a planting order fast."""

    def __init__(self, sites: pd.DataFrame, effects: pd.DataFrame, hexes: pd.DataFrame,
                 area: str, saturation_c: float = 1.5):
        s = sites[in_area(sites, area)].reset_index(drop=True).copy()
        if s.empty:
            raise ValueError(f"no planting sites in {area!r}")
        self.area = area
        self.sites = s
        self.effects = effects[effects["site_id"].isin(s["site_id"])]
        tgt_hex = pd.Index(self.effects["target_hex"].unique())
        self.hexes = hexes[hexes["hex"].isin(tgt_hex)].reset_index(drop=True)
        self.saturation_c = saturation_c
        self.pos = {sid: i for i, sid in enumerate(s["site_id"])}
        self._tgt, self._mag, *_ = _effect_vectors(self.effects, s["site_id"], self.hexes["hex"])
        self._w = {"cooling": self.hexes["residents"].to_numpy(float),
                   "vulnerability": self.hexes["vulnerable"].to_numpy(float)}
        # a first tree's effect, for coaching: °C on its own cell, and residents in the cells it cools
        res = self._w["cooling"]
        hpos = {h: i for i, h in enumerate(self.hexes["hex"])}
        self._own = [hpos.get(h, -1) for h in s["hex"]]   # index of each site's own cell in self.hexes
        self.sites["reach"] = [float(np.sum(m * res[t])) if t.size else 0.0
                               for t, m in zip(self._tgt, self._mag)]
        self.sites["first_c"] = [self._own_drop(i, m) for i, m in enumerate(self._mag)]
        self.sites["people_reached"] = [float(res[t].sum()) if t.size else 0.0 for t in self._tgt]

    def _own_drop(self, i: int, d: np.ndarray) -> float:
        """Cooling (°C) that effect vector d puts on site i's own cell."""
        hit = d[self._tgt[i] == self._own[i]]
        return float(hit.sum()) if hit.size else 0.0

    # ---- rules ---------------------------------------------------------------------------
    def costs(self, rnd: Round) -> np.ndarray:
        paved = self.sites["imperv_pct_2021"].fillna(0).to_numpy() > rnd.paved_pct
        return np.where(paved, rnd.cost_paved, rnd.cost_open).astype(float)

    def cost_of(self, site_id: Any, rnd: Round) -> float:
        return float(self.costs(rnd)[self.pos[site_id]])

    def capacity_of(self, site_id: Any) -> int:
        return int(self.sites.at[self.pos[site_id], "capacity"])

    def can_plant(self, order: list[Any], site_id: Any, rnd: Round) -> tuple[bool, str]:
        if site_id not in self.pos:
            return False, "Not a planting site."
        if order.count(site_id) >= self.capacity_of(site_id):
            return False, "This cell is full: no open ground left for another tree."
        if rnd.n_trees is not None and len(order) >= rnd.n_trees:
            return False, "No trees left. Lock in your plan."
        if rnd.budget_usd is not None:
            spent = sum(self.cost_of(s, rnd) for s in order)
            if spent + self.cost_of(site_id, rnd) > rnd.budget_usd:
                return False, "Not enough budget left for a tree here."
        return True, ""

    # ---- scoring -------------------------------------------------------------------------
    def evaluate(self, order: list[Any], rnd: Round) -> Outcome:
        w = self._w[rnd.objective]
        cooled = np.zeros(len(self.hexes))
        per_tree, per_c, spent = [], [], 0.0
        costs = self.costs(rnd)
        for sid in order:
            i = self.pos.get(sid)
            if i is None:
                per_tree.append(0.0)
                per_c.append(0.0)
                continue
            t, m = self._tgt[i], self._mag[i]
            if t.size:
                f = np.exp(-cooled[t] / self.saturation_c)
                d = m * f
                per_tree.append(float(np.sum(d * w[t])))
                per_c.append(self._own_drop(i, d))
                cooled[t] += d
            else:
                per_tree.append(0.0)
                per_c.append(0.0)
            spent += costs[i]
        ben = cooled >= BENEFIT_THRESHOLD_C
        hd = pd.DataFrame({"hex": self.hexes["hex"], "d_temp_c": -cooled})
        counts = pd.Series(order, dtype=object).value_counts().to_dict() if order else {}
        res_b = self.hexes["residents"].to_numpy(float)[ben]
        mean_c = float(np.average(cooled[ben], weights=res_b)) if res_b.sum() > 0 else 0.0
        return Outcome(per_tree_c=per_c, peak_c=float(cooled.max()) if cooled.size else 0.0, mean_c=mean_c,
                       score=float(np.sum(cooled * w)),
                       residents=int(self.hexes.loc[ben, "residents"].sum()),
                       vulnerable=int(self.hexes.loc[ben, "vulnerable"].sum()),
                       cells_cooled=int(ben.sum()), spent=float(spent), trees=len(order),
                       per_tree=per_tree, hex_delta=hd[hd.d_temp_c < 0].reset_index(drop=True),
                       counts=counts)

    def done(self, order: list[Any], rnd: Round) -> bool:
        """True when the round's trees are used up, or no site is affordable with what's left."""
        if rnd.n_trees is not None:
            return len(order) >= rnd.n_trees
        spent = sum(self.cost_of(s, rnd) for s in order)
        return bool(rnd.budget_usd - spent < self.costs(rnd).min())

    def next_tree(self, order: list[Any], site_id: Any) -> dict[str, float]:
        """What one more tree at site_id would do now: °C on its own cell (after diminishing returns),
        the biggest drop anywhere, and how many residents live in the cells it cools."""
        i = self.pos[site_id]
        t, m = self._tgt[i], self._mag[i]
        if not t.size:
            return {"own_c": 0.0, "peak_c": 0.0, "people": 0.0}
        cooled = np.zeros(len(self.hexes))
        for sid in order:
            j = self.pos.get(sid)
            if j is not None and self._tgt[j].size:
                tj = self._tgt[j]
                cooled[tj] += self._mag[j] * np.exp(-cooled[tj] / self.saturation_c)
        d = m * np.exp(-cooled[t] / self.saturation_c)
        return {"own_c": self._own_drop(i, d), "peak_c": float(d.max()),
                "people": float(self._w["cooling"][t].sum())}

    def marginal(self, order: list[Any], site_id: Any, rnd: Round) -> float:
        """Score the next tree at site_id would add (live feedback while planting)."""
        return self.evaluate(order + [site_id], rnd).score - self.evaluate(order, rnd).score

    # ---- references ----------------------------------------------------------------------
    def optimal_order(self, rnd: Round) -> list[Any]:
        s = self.sites.assign(cost=self.costs(rnd))
        c = PlanConstraints(objective=rnd.objective, n_trees=rnd.n_trees or 10_000,
                            budget_usd=rnd.budget_usd, cost_per_tree=rnd.cost_open)
        res = optimize(s, self.effects, self.hexes, c, rnd.cost_open, self.saturation_c)
        return res.picks["site_id"].tolist()

    def baseline(self, rnd: Round, strategy: str, seed: int = 7) -> list[Any]:
        """Random / hottest-first orders, cut at the round's tree count or budget."""
        full = baseline_order(self.sites, int(self.sites["capacity"].sum()), strategy, seed)
        out, spent = [], 0.0
        for sid in full:
            c = self.cost_of(sid, rnd)
            if rnd.n_trees is not None and len(out) >= rnd.n_trees:
                break
            if rnd.budget_usd is not None and spent + c > rnd.budget_usd:
                continue
            out.append(sid)
            spent += c
        return out


# ---- after the round ------------------------------------------------------------------------
def pct_of(player: Outcome, best: Outcome) -> float:
    return 100.0 * player.score / best.score if best.score > 0 else 0.0


def final_pct(player: Outcome, best: Outcome, intel: set[str]) -> float:
    return max(0.0, pct_of(player, best) - sum(INTEL_COST.get(i, 0) for i in intel))


def coaching(board: Board, order: list[Any], best_order: list[Any], player: Outcome,
             top: int = 3) -> dict[str, Any]:
    """The optimizer's best cells the player skipped, and the player's weakest tree."""
    s = board.sites.set_index("site_id")
    best_counts = pd.Series(best_order, dtype=object).value_counts()
    missed = [sid for sid in best_counts.index if sid not in set(order)][:top]
    worst = None
    if order:
        k = int(np.argmin(player.per_tree))
        worst = {"site_id": order[k], "gain": player.per_tree[k], "tree_no": k + 1,
                 "gain_c": player.per_tree_c[k] if player.per_tree_c else 0.0,
                 "repeat": order[:k].count(order[k])}
    return {"missed": [{"site_id": sid, "trees": int(best_counts[sid]), **_facts(s, sid)}
                       for sid in missed],
            "worst": None if worst is None else {**worst, **_facts(s, worst["site_id"])}}


def _facts(s: pd.DataFrame, sid: Any) -> dict[str, Any]:
    r = s.loc[sid]
    return {"residents": float(r.get("residents", 0) or 0), "reach": float(r.get("reach", 0) or 0),
            "heat_c": float(r.get("lst_anom_c", 0) or 0),
            "first_c": float(r.get("first_c", 0) or 0),
            "people_reached": float(r.get("people_reached", 0) or 0),
            "canopy_pct": float(r.get("canopy_pct_2021", 0) or 0),
            "imperv_pct": float(r.get("imperv_pct_2021", 0) or 0),
            "vulnerable": float(r.get("vulnerable", 0) or 0)}


def why(f: dict[str, Any], scale: float = 1.0, unit: str = "°C") -> str:
    """One plain-language reason a cell is (or isn't) a good spot. scale/unit: 1.8, "°F" for Fahrenheit."""
    bits = []
    if f.get("first_c", 0) > 0:
        bits.append(f"a first tree cools this block by {f['first_c'] * scale:.2f} {unit}"
                    + (f" and reaches about {f['people_reached']:,.0f} people nearby"
                       if f.get("people_reached", 0) >= 1 else ""))
    if f["residents"] >= 150:
        bits.append(f"{f['residents']:,.0f} residents live right here")
    elif f["residents"] < 30:
        bits.append("almost nobody lives here")
    if f["heat_c"] > 3:
        bits.append(f"{f['heat_c'] * scale:+.1f} {unit} hotter than the city median")
    if f["canopy_pct"] > 25:
        bits.append(f"already {f['canopy_pct']:.0f}% shaded")
    return "; ".join(bits) or "an average cell"


def achievements(rnd: Round, player: Outcome, best: Outcome, pct: float,
                 intel: set[str]) -> list[tuple[str, str]]:
    out = []
    if player.score >= best.score > 0:
        out.append(("Beat the algorithm", "You matched or beat the optimizer."))
    if player.cells_cooled >= max(15, int(0.8 * best.cells_cooled)):
        out.append(("Shade spreader", f"You cooled {player.cells_cooled} cells."))
    if player.counts and max(player.counts.values()) <= 3 and pct >= 70:
        out.append(("Light touch", "No cell got more than 3 trees, and it still scored well."))
    if rnd.budget_usd and player.spent <= 0.9 * rnd.budget_usd and pct >= 70:
        out.append(("Budget hawk", f"Grade B or better with ${rnd.budget_usd - player.spent:,.0f} left over."))
    if not intel and pct >= 70:
        out.append(("Gut instinct", "No paid hints, grade B or better."))
    if pct >= 50 and player.residents and player.vulnerable / player.residents >= 0.5:
        out.append(("Equity champion", "Over half the people you cooled are vulnerable."))
    return out
