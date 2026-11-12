"""Typed planning constraints: the single source of truth shared by the LLM, the UI chips
and the optimizer.

The LLM only *fills* a PlanConstraints object. Validation, coercion and the decision to
ask a clarifying question happen here in plain Python, because hosted models were seen to
(a) send lists as strings, numbers as strings, and (b) silently pick "hard" for an
ambiguous "prioritize District 3" (tool-call test, 2026-11-07).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Exclusion = Literal["overhead_wires", "transmission", "narrow_sidewalk"]
AttractorType = Literal["bus_stops", "schools", "senior_centers"]
Objective = Literal["cooling", "vulnerability", "balanced"]


def _maybe_json(v: Any) -> Any:
    """Models sometimes send '["a","b"]' or '{"k":1}' as a string; decode it."""
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in "[{":
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return v
    return v


def _as_list(v: Any) -> Any:
    v = _maybe_json(v)
    if v is None:
        return []
    if isinstance(v, (str, dict)):
        return [v]
    return v


_DISTRICT_CODE = re.compile(r"[A-Z0-9]{1,2}")
# words too generic to identify a Super Neighborhood on their own ("Greater Uptown" -> "uptown")
_AREA_STOPWORDS = {"GREATER", "AREA", "AREAS", "PARK", "NORTH", "SOUTH", "EAST", "WEST", "NORTHEAST",
                   "NORTHWEST", "SOUTHEAST", "SOUTHWEST", "CENTRAL", "HOUSTON", "WARD", "THE", "AND",
                   "OF", "TRI-COMMUNITY", "COMMUNITY", "GARDENS", "HEIGHTS", "PLACE", "ACRES", "VILLAGE"}


def is_district_code(area: str) -> bool:
    """Council districts are 1-2 character codes ("J", "3"); anything longer is a Super Neighborhood."""
    return bool(_DISTRICT_CODE.fullmatch(str(area).strip().upper()))


def area_label(area: str) -> str:
    return f"District {area}" if is_district_code(area) else str(area).title()


def area_pattern(area: str) -> re.Pattern:
    """How a user's text refers to an area: "district J" for districts; for a Super Neighborhood
    the full name, or any distinctive word of it ("Uptown" for "GREATER UPTOWN")."""
    a = str(area).strip()
    if is_district_code(a):
        return re.compile(rf"\bdistrict\s*#?\s*{re.escape(a)}\b", re.I)
    words = [w for w in re.split(r"[\s/,\-]+", a.upper()) if len(w) >= 4 and w not in _AREA_STOPWORDS]
    alts = [re.escape(a)] + [re.escape(w) for w in words]
    return re.compile(r"\b(" + "|".join(alts) + r")\b", re.I)


class DistrictPref(BaseModel):
    """A preference for an area: a council district ("C", "3") or a Super Neighborhood ("GULFTON").
    The field keeps its historical name so tool calls and saved state stay compatible."""
    district: str

    @field_validator("district", mode="before")
    @classmethod
    def _norm_district(cls, v: Any) -> str:
        s = str(v).strip()
        s = re.sub(r"^(council\s+)?district\s*", "", s, flags=re.I)
        return s.upper()

    # None = the user did not say; the agent must ask before optimizing.
    strength: Literal["soft", "hard"] | None = None
    # For hard prefs: minimum share of trees in this district (0-1).
    min_share: float | None = Field(default=None, ge=0, le=1)
    # For soft prefs: multiplier on benefit score for sites in this district.
    weight: float = Field(default=2.0, gt=0)

    @property
    def is_district(self) -> bool:
        return is_district_code(self.district)

    @property
    def label(self) -> str:
        return area_label(self.district)


class Attractor(BaseModel):
    type: AttractorType
    radius_m: float = Field(default=100, gt=0)
    weight: float = Field(default=1.5, gt=0)


class PlanConstraints(BaseModel):
    budget_usd: float | None = Field(default=None, gt=0)
    n_trees: int | None = Field(default=None, gt=0)
    cost_per_tree: float | None = Field(default=None, gt=0)
    exclude: list[Exclusion] = Field(default_factory=list)
    district_prefs: list[DistrictPref] = Field(default_factory=list)
    attractors: list[Attractor] = Field(default_factory=list)
    objective: Objective = "vulnerability"

    # One before-validator per field: pydantic runs stacked before-validators in reverse
    # definition order, so splitting coercion across several is fragile.
    @field_validator("district_prefs", mode="before")
    @classmethod
    def _coerce_prefs(cls, v: Any) -> Any:
        return _as_list(v)

    @field_validator("attractors", mode="before")
    @classmethod
    def _coerce_attractors(cls, v: Any) -> Any:
        return [{"type": a} if isinstance(a, str) else a for a in _as_list(v)]

    @field_validator("exclude", mode="before")
    @classmethod
    def _coerce_exclude(cls, v: Any) -> Any:
        alias = {"power_lines": "overhead_wires", "wires": "overhead_wires",
                 "power lines": "overhead_wires"}
        return list(dict.fromkeys(alias.get(x, x) if isinstance(x, str) else x
                                  for x in _as_list(v)))

    @model_validator(mode="after")
    def _hard_needs_share(self) -> "PlanConstraints":
        for p in self.district_prefs:
            if p.strength == "hard" and p.min_share is None:
                p.min_share = 0.4  # default quota; surfaced as an editable chip
        return self

    # ---- helpers -------------------------------------------------------------------
    def effective_cost(self, default_cost: float) -> float:
        return self.cost_per_tree or default_cost

    def resolved_n_trees(self, default_cost: float) -> int:
        """Trees to plant: explicit n_trees, capped by the budget if both are given."""
        cost = self.effective_cost(default_cost)
        by_budget = math.floor(self.budget_usd / cost) if self.budget_usd else None
        if self.n_trees and by_budget is not None:
            return min(self.n_trees, by_budget)
        return self.n_trees or by_budget or 0

    def chips(self) -> list[str]:
        """Human-readable chips for the left panel of the app."""
        out: list[str] = []
        if self.budget_usd:
            out.append(f"Budget ${self.budget_usd:,.0f}")
        if self.n_trees:
            out.append(f"{self.n_trees:,} trees")
        if self.cost_per_tree:
            out.append(f"${self.cost_per_tree:,.0f}/tree")
        out += [f"Exclude: {e.replace('_', ' ')}" for e in self.exclude]
        for p in self.district_prefs:
            if p.strength == "hard":
                out.append(f"{p.label}: ≥{p.min_share:.0%} of trees")
            elif p.strength == "soft":
                out.append(f"{p.label}: preferred (×{p.weight:g})")
            else:
                out.append(f"{p.label}: priority (hard or soft?)")
        out += [f"Favor: {a.type.replace('_', ' ')} (≤{a.radius_m:g} m)" for a in self.attractors]
        out.append(f"Objective: {self.objective}")
        return out


# ---- clarification rules ---------------------------------------------------------------

_HARD = re.compile(r"\b(at least|minimum|min\.?|must|only|require[sd]?|requirement|"
                   r"no less than|guarantee[sd]?|quota|hard)\b|\d+\s?%", re.I)
_SOFT = re.compile(r"\b(prefer(ably|red|ence)?|favou?r|lean toward|if possible|ideally|"
                   r"nice to have|soft|tilt)\b", re.I)


@dataclass
class Clarification:
    field: str
    question: str
    options: list[str] = field(default_factory=list)


_CLAUSE_SPLIT = re.compile(r"[.;,]|\band\b|\bbut\b", re.I)


def strength_stated(user_text: str, district: str | None = None
                    ) -> Literal["soft", "hard"] | None:
    """Did the user's own words say hard vs soft? Words like 'prioritize' say neither.

    With `district`, only the clause(s) that mention that district count, so
    "favor bus stops" does not make "prioritize District 3" soft.
    """
    text = user_text
    if district is not None:
        pat = area_pattern(str(district))
        parts = _CLAUSE_SPLIT.split(user_text)
        clauses = []
        for k, cl in enumerate(parts):
            if pat.search(cl):
                clauses.append(cl)
                # a bare list item ("prefer Gulfton and Sharpstown") takes the previous clause's verb
                if k and not re.sub(r"\b(in|the|also)\b", "", pat.sub("", cl), flags=re.I).strip():
                    clauses.append(parts[k - 1])
        text = " ".join(clauses)
    if _HARD.search(text):
        return "hard"
    if _SOFT.search(text):
        return "soft"
    return None


def reconcile(c: PlanConstraints, user_text: str) -> PlanConstraints:
    """Override LLM-guessed district strength with what the text actually supports."""
    for p in c.district_prefs:
        stated = strength_stated(user_text, p.district)
        p.strength = stated
        if stated != "hard":
            p.min_share = None
    return c


def clarifications(c: PlanConstraints, default_cost: float,
                   cost_tolerance: float = 0.35) -> list[Clarification]:
    qs: list[Clarification] = []
    if not c.budget_usd and not c.n_trees:
        qs.append(Clarification("n_trees", "How many trees, or what budget, should I plan for?"))
    for p in c.district_prefs:
        if p.strength is None:
            qs.append(Clarification(
                f"district_prefs[{p.district}].strength",
                f"Should {p.label} be a hard requirement (e.g. at least 40% of "
                f"trees) or a strong preference?",
                ["Hard: at least 40%", "Strong preference"]))
    if c.budget_usd and c.n_trees and not c.cost_per_tree:
        implied = c.budget_usd / c.n_trees
        if abs(implied - default_cost) / default_cost > cost_tolerance:
            fit = math.floor(c.budget_usd / default_cost)
            qs.append(Clarification(
                "cost_per_tree",
                f"${c.budget_usd:,.0f} for {c.n_trees:,} trees is ${implied:,.0f}/tree, but the "
                f"cost table says ${default_cost:,.0f}/tree (≈{fit:,} trees). Which should I use?",
                [f"Use ${implied:,.0f}/tree", f"Plan {fit:,} trees"]))
    return qs
