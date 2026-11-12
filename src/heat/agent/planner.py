"""Planner agent: tool-calling loop over the optimizer + Unity Catalog functions.

Design rules (from the plan):
  * The LLM never owns the plan. It fills a PlanConstraints object; Python validates it,
    decides whether to ask a clarifying question, runs the optimizer and returns numbers.
  * Hard vs soft district preference comes from the user's words (or their answer to our
    question, or a chip edit), never from the model's guess.
  * Every number in the reply should come from a tool result; we check and flag the rest.
  * Each optimize call also runs the sibling objective and offers it when the trade-off is
    big enough (rule-based, so the agent can't invent trade-offs).
Works with any OpenAI-compatible chat client (Databricks serving endpoints are).
"""
from __future__ import annotations

import difflib
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from heat.agent.constraints import (PlanConstraints, area_label, area_pattern, clarifications,
                                    is_district_code, strength_stated)
from heat.sim.optimize import PlanResult, in_area, optimize

try:  # tracing is optional (local tests run without MLflow)
    import mlflow
    trace = mlflow.trace
except Exception:  # pragma: no cover
    def trace(fn=None, **_):
        return fn if fn else (lambda f: f)

C_TO_F = 1.8

SYSTEM_PROMPT = """You are a tree-planting planner for the City of Houston's urban forestry team.
You help a manager turn a planting request (budget and/or number of trees, places to avoid,
districts to prioritize, what to optimize for) into an optimized plan, and answer questions about it.
If the request has no budget and no tree count, ask for one instead of calling optimize_plan.

Rules:
- Use tools for every number. Never invent or estimate figures yourself.
- To plan, call optimize_plan with the constraints the user stated. Council districts are letters A-K.
  Neighborhood names (Houston's Super Neighborhoods, e.g. Gulfton, Sharpstown, Second Ward) go in
  district_prefs too: put the neighborhood name in the `district` field.
- If a tool returns needs_clarification, ask exactly those questions (briefly) and stop.
- Temperatures are SURFACE temperatures from Landsat (~10:30am summer). Report cooling in deg F
  using the *_f fields, and give the range when provided.
- If a tool result includes `alternative`, mention it in one sentence and offer to compare.
- If a tool result includes `unsupported`, tell the user plainly that those constraints could not
  be applied because the data is not loaded. Never claim the plan honors them: do not describe
  the plan as avoiding, favoring or being near anything listed in `unsupported`.
- Only pass constraints the user actually stated. Do not add exclusions or preferences on your own.
- explain_site needs an H3 cell id or a latitude/longitude. Street names are not loaded; if the user
  names a street, ask for a map point (or a cell id from the table).
- For "why not <place>?" call explain_site. For "what if the budget goes up 20%?" call what_if.
- Keep answers short: 2-5 sentences, plain language, no tables unless asked."""


def _tool(name: str, description: str, properties: dict[str, Any],
          required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required or []}}}


TOOLS = [
    _tool("optimize_plan", "Run the planting optimizer with the user's constraints.", {
        "budget_usd": {"type": "number"}, "n_trees": {"type": "integer"},
        "cost_per_tree": {"type": "number"},
        "exclude": {"type": "array", "items": {"type": "string",
                    "enum": ["overhead_wires", "transmission", "narrow_sidewalk"]}},
        "district_prefs": {"type": "array", "items": {"type": "object", "properties": {
            "district": {"type": "string",
                         "description": "Council district letter A-K, or a Super Neighborhood name"},
            "strength": {"type": "string", "enum": ["soft", "hard"]},
            "min_share": {"type": "number"}}, "required": ["district"]}},
        "attractors": {"type": "array", "items": {"type": "string",
                       "enum": ["bus_stops", "schools", "senior_centers"]}},
        "objective": {"type": "string", "enum": ["cooling", "vulnerability", "balanced"]}}),
    _tool("what_if", "Re-run the current plan with a change, e.g. budget +20% or more trees.", {
        "budget_multiplier": {"type": "number"}, "n_trees": {"type": "integer"},
        "budget_usd": {"type": "number"},
        "objective": {"type": "string", "enum": ["cooling", "vulnerability", "balanced"]}}),
    _tool("explain_site", "Explain why a cell was or wasn't chosen in the current plan. "
          "Pass an H3 cell id, or lat and lon.",
          {"h3_cell": {"type": "string"}, "lat": {"type": "number"}, "lon": {"type": "number"}}),
    _tool("compare_plans", "Compare two plans by id (from earlier optimize/what_if results).",
          {"plan_a": {"type": "string"}, "plan_b": {"type": "string"}}, ["plan_a", "plan_b"]),
    _tool("district_summary", "Heat, canopy and population per council district (A-K or ALL).",
          {"district_id": {"type": "string"}}, ["district_id"]),
    _tool("neighborhood_summary", "Heat, canopy and population per Super Neighborhood (Houston's 88 "
          "neighborhood planning areas). Pass a name or part of one (e.g. Gulfton), or ALL.",
          {"name": {"type": "string"}}, ["name"]),
    _tool("hottest_cells", "Hottest populated cells, filterable by council district letter or Super "
          "Neighborhood name, elderly residents, canopy.", {
        "district_id": {"type": "string"}, "min_age65": {"type": "number"},
        "max_canopy_pct": {"type": "number"}, "n": {"type": "integer"}}, ["district_id"]),
    _tool("site_details", "Facts about one H3 cell (heat, canopy, residents, SVI).",
          {"h3_cell": {"type": "string"}}, ["h3_cell"]),
    _tool("model_card", "Heat model accuracy and headline effect sizes.", {}),
]

UC_FUNCTIONS = {"district_summary", "neighborhood_summary", "hottest_cells", "site_details", "model_card"}


@dataclass
class PlannerData:
    sites: pd.DataFrame      # site_id, hex, district, capacity, cost?, lst_anom_c, flags...
    effects: pd.DataFrame    # site_id, target_hex, d_temp_c, d_temp_lo, d_temp_hi
    hexes: pd.DataFrame      # hex, residents, vulnerable
    default_cost: float = 1500.0
    saturation_c: float = 1.5
    cost_tolerance: float = 0.35


@dataclass
class PlannerState:
    constraints: PlanConstraints | None = None
    pending_districts: list[str] = field(default_factory=list)   # asked hard/soft, awaiting answer
    resolved_strength: dict[str, str] = field(default_factory=dict)
    plans: dict[str, dict[str, Any]] = field(default_factory=dict)  # plan_id -> summary
    results: dict[str, PlanResult] = field(default_factory=dict)
    current_plan: str | None = None


def h3_str(v: Any) -> str:
    return format(int(v), "x") if not isinstance(v, str) else v.lower()


class Planner:
    def __init__(self, data: PlannerData, llm: Any, model: str,
                 uc_call: Callable[[str, dict[str, Any]], Any] | None = None,
                 max_steps: int = 6):
        self.d, self.llm, self.model, self.uc_call = data, llm, model, uc_call
        self.max_steps = max_steps
        self.state = PlannerState()
        self._sites_by_h3 = {h3_str(s): s for s in data.sites["site_id"]}
        self._nbhds = (sorted(data.sites["super_nbhd"].dropna().str.upper().unique())
                       if "super_nbhd" in data.sites else [])

    def resolve_neighborhood(self, name: str) -> str | None:
        """Map what the user/LLM wrote ("Gulfton", "uptown", "2nd ward") to a Super Neighborhood
        name in the data (upper case), or None if nothing matches well enough."""
        q = re.sub(r"\s+", " ", str(name).strip().upper())
        q = re.sub(r"\b(SUPER\s*NEIGHBORHOOD|NEIGHBORHOOD)\b", "", q).strip()
        if not q or not self._nbhds:
            return None
        if q in self._nbhds:
            return q
        hits = [n for n in self._nbhds if area_pattern(n).search(q)] or \
               [n for n in self._nbhds if q in n]
        if len(hits) == 1:
            return hits[0]
        close = difflib.get_close_matches(q, hits or self._nbhds, n=1, cutoff=0.75)
        return close[0] if close else None

    # ---- grounding the constraints in the user's words -----------------------------------
    _EVIDENCE = {"overhead_wires": r"power|wire|utility|line", "transmission": r"transmission|power",
                 "narrow_sidewalk": r"sidewalk|narrow", "bus_stops": r"\bbus|transit|metro\b",
                 "schools": r"school", "senior_centers": r"senior|elder|older"}

    def _grounded(self, c: PlanConstraints, all_user_text: str) -> PlanConstraints:
        """Drop constraints the user never stated (the LLM sometimes adds them, e.g. copying
        from examples): exclusions/attractors need a keyword, districts need "district X",
        and budget / tree counts / cost need the number to appear in the user's words."""
        ok = lambda k: re.search(self._EVIDENCE[k], all_user_text, re.I) is not None
        c.exclude = [e for e in c.exclude if ok(e)]
        c.attractors = [a for a in c.attractors if ok(a.type)]
        kept = []
        for p in c.district_prefs:
            if is_district_code(p.district):
                if re.search(rf"\bdistricts?\b[^.;]*?\b{re.escape(p.district)}\b", all_user_text, re.I):
                    kept.append(p)
                continue
            # a neighborhood: the user must have named it; then use the name as the data spells it
            if area_pattern(p.district).search(all_user_text):
                p.district = self.resolve_neighborhood(p.district) or p.district
                kept.append(p)
        c.district_prefs = kept
        nums = user_numbers(all_user_text)
        near = lambda v: any(abs(v - n) <= max(1.0, 0.005 * n) for n in nums)
        for k in ("budget_usd", "n_trees", "cost_per_tree"):
            v = getattr(c, k)
            if v is not None and not near(float(v)):
                setattr(c, k, None)
        return c

    def unsupported(self, c: PlanConstraints) -> list[str]:
        """Constraints that can't change the plan because their data layer isn't loaded."""
        s, out = self.d.sites, []
        for e in c.exclude:
            if e not in s or not s[e].fillna(False).astype(bool).any():
                out.append(f"exclude {e}: no {e.replace('_', ' ')} data loaded for this city")
        for a in c.attractors:
            if f"dist_{a.type}_m" not in s:
                out.append(f"favor {a.type}: no {a.type.replace('_', ' ')} locations loaded")
        for p in c.district_prefs:
            if not in_area(s, p.district).any():
                kind = "council district" if is_district_code(p.district) else "Super Neighborhood"
                out.append(f"{area_label(p.district)}: no {kind} by that name has planting sites "
                           f"in the data, so this preference has no effect")
        return out

    # ---- strength resolution ------------------------------------------------------------
    def _resolve(self, c: PlanConstraints, user_text: str) -> PlanConstraints:
        for p in c.district_prefs:
            said = strength_stated(user_text, p.district)
            if said is None and p.district in self.state.pending_districts:
                said = strength_stated(user_text)          # answer to our question
            if said is None:
                said = self.state.resolved_strength.get(p.district)
            p.strength = said
            if said == "hard" and p.min_share is None:
                m = re.search(r"(\d{1,2})\s?%", user_text)
                p.min_share = int(m.group(1)) / 100 if m else 0.4
            if said != "hard":
                p.min_share = None
            if said:
                self.state.resolved_strength[p.district] = said
        return c

    # ---- tools ------------------------------------------------------------------------
    def _run_plan(self, c: PlanConstraints, label: str | None = None) -> dict[str, Any]:
        qs = clarifications(c, self.d.default_cost, self.d.cost_tolerance)
        self.state.constraints = c
        if qs:
            self.state.pending_districts = [q.field.split("[")[1].split("]")[0]
                                            for q in qs if q.field.startswith("district_prefs")]
            return {"needs_clarification": [{"field": q.field, "question": q.question,
                                             "options": q.options} for q in qs],
                    "chips": c.chips()}
        self.state.pending_districts = []
        res = optimize(self.d.sites, self.d.effects, self.d.hexes, c, self.d.default_cost,
                       self.d.saturation_c)
        pid = label or f"plan_{len(self.state.plans) + 1}_{uuid.uuid4().hex[:4]}"
        out = self._summarize(pid, c, res)
        if self.unsupported(c):
            out["unsupported"] = self.unsupported(c)
        self.state.results[pid], self.state.plans[pid] = res, out
        self.state.current_plan = pid
        alt = self._alternative(c, res)
        if alt:
            out["alternative"] = alt
        return out

    def _summarize(self, pid: str, c: PlanConstraints, res: PlanResult) -> dict[str, Any]:
        s = dict(res.summary)
        mean_c = s.get("mean_cooling_benefiting_hexes_c", 0.0)
        out = {"plan_id": pid, "constraints": c.model_dump(), "chips": c.chips(),
               "n_trees": s["n_trees_planned"], "cost_usd": s["cost_usd"],
               "residents_benefiting": s["residents_benefiting"],
               "vulnerable_benefiting": s["vulnerable_benefiting"],
               "mean_cooling_c": mean_c, "mean_cooling_f": round(mean_c * C_TO_F, 2),
               "trees_by_district": s.get("trees_by_district", {})}
        if c.district_prefs and len(res.picks):   # how many trees landed in each named area
            picked = self.d.sites.set_index("site_id").loc[res.picks["site_id"]].reset_index()
            out["trees_in_named_areas"] = {area_label(p.district): int(in_area(picked, p.district).sum())
                                           for p in c.district_prefs}
        if "mean_cooling_range_c" in s:
            lo, hi = s["mean_cooling_range_c"]
            out["mean_cooling_range_f"] = [round(lo * C_TO_F, 2), round(hi * C_TO_F, 2)]
        top = res.picks.groupby("site_id").size().sort_values(ascending=False).head(5)
        out["top_sites"] = [{"h3_cell": h3_str(k), "trees": int(v)} for k, v in top.items()]
        # trade-off: hottest candidate cells citywide that this plan skipped, and why
        hot = self.d.sites.sort_values("lst_anom_c", ascending=False).head(10)
        log = res.decision_log.set_index("site_id")
        skipped = [{"h3_cell": h3_str(i), "reason": log.at[i, "reason"]}
                   for i in hot["site_id"] if i in log.index and log.at[i, "trees"] == 0]
        out["hottest10_skipped"] = skipped
        return out

    def _alternative(self, c: PlanConstraints, res: PlanResult) -> dict[str, Any] | None:
        other = "cooling" if c.objective == "vulnerability" else "vulnerability"
        c2 = c.model_copy(deep=True)
        c2.objective = other
        r2 = optimize(self.d.sites, self.d.effects, self.d.hexes, c2, self.d.default_cost,
                      self.d.saturation_c)
        a, b = res.summary, r2.summary
        if not a["vulnerable_benefiting"] or not a["degree_person_c"]:
            return None
        dv = b["vulnerable_benefiting"] / a["vulnerable_benefiting"] - 1
        dc = b["degree_person_c"] / a["degree_person_c"] - 1
        if (dv >= 0.10 and dc >= -0.05) or (dc >= 0.10 and dv >= -0.05):
            pid = f"alt_{other}_{uuid.uuid4().hex[:4]}"
            summ = self._summarize(pid, c2, r2)
            self.state.results[pid], self.state.plans[pid] = r2, summ
            return {"plan_id": pid, "objective": other,
                    "vulnerable_benefiting_change_pct": round(100 * dv, 1),
                    "total_cooling_change_pct": round(100 * dc, 1)}
        return None

    def _explain(self, h3_cell: str, lat: float | None = None,
                 lon: float | None = None) -> dict[str, Any]:
        pid = self.state.current_plan
        if not pid:
            return {"error": "No plan yet. Run optimize_plan first."}
        if (lat is not None and lon is not None) and self.uc_call:
            got = self.uc_call("nearest_cell", {"lat": float(lat), "lon": float(lon)})
            h3_cell = (got[0].get("h3_cell") if got else "") or ""
        key = h3_str(h3_cell) if h3_cell else ""
        if key not in self._sites_by_h3 and not re.fullmatch(r"[0-9a-f]{15}", key):
            return {"error": "That isn't a map location I can look up. Street names are not "
                             "loaded; ask the user for a point on the map or a cell id."}
        sid = self._sites_by_h3.get(key)
        if sid is None:
            return {"h3_cell": h3_cell, "status": "not_a_candidate",
                    "reason": "Not a planting candidate (water, already >=40% canopy, "
                              "<10% / >95% impervious, or no open ground left once buildings "
                              "and roads are excluded)."}
        row = self.state.results[pid].decision_log.set_index("site_id").loc[sid]
        out = {"plan_id": pid, "h3_cell": h3_str(sid), "status": row["status"],
               "reason": row["reason"], "trees": int(row["trees"])}
        if row["reason"] == "outranked":
            out["benefit_vs_cutoff"] = round(float(row["gain_vs_cutoff"]), 2)
            out["note"] = "Its benefit per tree was below the last tree the plan accepted."
        return out

    def _what_if(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self.state.constraints:
            return {"error": "No plan yet. Run optimize_plan first."}
        c = self.state.constraints.model_copy(deep=True)
        if args.get("budget_multiplier") and c.budget_usd:
            c.budget_usd = c.budget_usd * float(args["budget_multiplier"])
            if c.n_trees:
                c.n_trees = int(round(c.n_trees * float(args["budget_multiplier"])))
        for k in ("n_trees", "budget_usd", "objective"):
            if args.get(k) is not None:
                setattr(c, k, args[k])
        base = self.state.current_plan
        out = self._run_plan(c)
        if base and "plan_id" in out:
            out["vs_previous"] = self._compare(base, out["plan_id"])
        return out

    def _compare(self, a: str, b: str) -> dict[str, Any]:
        pa, pb = self.state.plans.get(a), self.state.plans.get(b)
        if not pa or not pb:
            return {"error": f"Unknown plan id(s). Known: {list(self.state.plans)}"}
        keys = ["n_trees", "cost_usd", "residents_benefiting", "vulnerable_benefiting",
                "mean_cooling_f"]
        return {"plan_a": a, "plan_b": b,
                **{k: {"a": pa[k], "b": pb[k], "diff": round(pb[k] - pa[k], 2)} for k in keys}}

    @trace(name="tool_call", span_type="TOOL")
    def call_tool(self, name: str, args: dict[str, Any], user_text: str,
                  all_user_text: str = "") -> Any:
        if name == "optimize_plan":
            try:
                c = PlanConstraints.model_validate(args)
            except Exception as e:  # invalid even after coercion
                return {"error": f"Invalid constraints: {e}"}
            c = self._grounded(c, all_user_text or user_text)
            return self._run_plan(self._resolve(c, user_text))
        if name == "what_if":
            return self._what_if(args)
        if name == "explain_site":
            return self._explain(args.get("h3_cell", ""), args.get("lat"), args.get("lon"))
        if name == "compare_plans":
            return self._compare(args.get("plan_a", ""), args.get("plan_b", ""))
        if name in UC_FUNCTIONS:
            if not self.uc_call:
                return {"error": "UC functions unavailable in this environment"}
            if name == "hottest_cells":
                args = {"district_id": args.get("district_id", "ALL"),
                        "min_age65": float(args.get("min_age65", 0) or 0),
                        "max_canopy_pct": float(args.get("max_canopy_pct", 100) or 100),
                        "n": int(args.get("n", 10) or 10)}
            if name == "neighborhood_summary":
                args = {"name": str(args.get("name") or args.get("district_id") or "ALL")}
            return self.uc_call(name, args)
        return {"error": f"unknown tool {name}"}

    # ---- chat loop --------------------------------------------------------------------
    def apply_chip_edit(self, c: PlanConstraints) -> None:
        """UI edits win: they become the resolved state the next run starts from."""
        self.state.constraints = c
        for p in c.district_prefs:
            if p.strength:
                self.state.resolved_strength[p.district] = p.strength

    @trace(name="planner_chat", span_type="AGENT")
    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        user_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        all_user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
        convo = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.state.constraints:
            convo.append({"role": "system", "content": "Current constraints: "
                          + json.dumps(self.state.constraints.model_dump())})
        convo += messages
        tool_log: list[dict[str, Any]] = []
        for _ in range(self.max_steps):
            resp = self.llm.chat.completions.create(model=self.model, messages=convo,
                                                    tools=TOOLS, max_tokens=700, temperature=0)
            msg = resp.choices[0].message
            calls = getattr(msg, "tool_calls", None) or []
            if not calls:
                reply = msg.content or ""
                return {"reply": reply, "tool_log": tool_log,
                        "ungrounded_numbers": ungrounded_numbers(reply, tool_log, user_text),
                        "constraints": (self.state.constraints.model_dump()
                                        if self.state.constraints else None),
                        "chips": self.state.constraints.chips() if self.state.constraints else [],
                        "plan_id": self.state.current_plan}
            convo.append({"role": "assistant", "content": msg.content or "",
                          "tool_calls": [{"id": tc.id, "type": "function",
                                          "function": {"name": tc.function.name,
                                                       "arguments": tc.function.arguments}}
                                         for tc in calls]})
            for tc in calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self.call_tool(tc.function.name, args, user_text, all_user_text)
                tool_log.append({"tool": tc.function.name, "args": args, "result": result})
                convo.append({"role": "tool", "tool_call_id": tc.id,
                              "content": json.dumps(result, default=str)[:12000]})
        return {"reply": "I couldn't finish that within the step limit.", "tool_log": tool_log,
                "ungrounded_numbers": [], "constraints": None, "chips": [], "plan_id": None}


_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*")
_MONEY = re.compile(r"(\d[\d,]*\.?\d*)\s*(k|thousand|m|mm|million|b|billion)?\b", re.I)
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}


def user_numbers(text: str) -> list[float]:
    """Numbers a user wrote, with k / million suffixes expanded ("$250k" -> 250000)."""
    out = []
    for num, suf in _MONEY.findall(text):
        try:
            v = float(num.replace(",", ""))
        except ValueError:
            continue
        out.append(v * _MULT.get((suf or "").lower(), 1.0))
    return out


def ungrounded_numbers(reply: str, tool_log: list[dict[str, Any]], user_text: str) -> list[str]:
    """Numbers in the reply that appear in no tool result (nor the user's message)."""
    source = json.dumps([t["result"] for t in tool_log], default=str) + " " + user_text
    have = set()
    for tok in _NUM.findall(source):
        try:
            v = float(tok.replace(",", ""))
        except ValueError:
            continue
        have |= {round(v, 0), round(v, 1), round(v, 2), round(abs(v), 0), round(abs(v), 1),
                 round(abs(v), 2)}
        if 0 < abs(v) <= 1:
            have.add(round(abs(v) * 100, 0))       # shares quoted as percents
    out = []
    for tok in _NUM.findall(reply):
        try:
            v = float(tok.replace(",", "").rstrip("."))
        except ValueError:
            continue
        if v in (0, 1, 2, 3, 4, 5, 10) or (1900 <= v <= 2100):
            continue                                 # counting words, years
        if not ({round(v, 0), round(v, 1), round(v, 2)} & have):
            out.append(tok)
    return out
