"""Planner loop with a scripted fake LLM: checks the clarification flow, strength
resolution across turns, explain_site and what_if without needing a model endpoint."""
import json
from types import SimpleNamespace as NS

from heat.agent.planner import Planner, PlannerData, ungrounded_numbers
from tests.test_optimize import toy


class FakeLLM:
    """Returns queued responses: ("tool", name, args) or ("text", content)."""
    def __init__(self, script):
        self.script = list(script)
        self.chat = NS(completions=NS(create=self._create))
        self.seen = []

    def _create(self, **kw):
        self.seen.append(kw["messages"])
        kind, *rest = self.script.pop(0)
        if kind == "tool":
            name, args = rest
            tc = NS(id=f"c{len(self.seen)}", function=NS(name=name, arguments=json.dumps(args)))
            msg = NS(content="", tool_calls=[tc])
        else:
            msg = NS(content=rest[0], tool_calls=None)
        return NS(choices=[NS(message=msg)])


def data():
    sites, eff, hexes = toy()
    sites["lst_anom_c"] = [3.0, 2.0, 1.0, 0.5, 4.0]
    return PlannerData(sites, eff, hexes, default_cost=1000)


def last_tool_result(llm):
    return json.loads(llm.seen[-1][-1]["content"])


def test_ambiguous_district_asks_then_resolves_on_answer():
    ask = "We have $6,000 for 6 trees. Avoid power lines and prioritize District C."
    llm = FakeLLM([
        ("tool", "optimize_plan", {"budget_usd": 6000, "n_trees": 6, "exclude": "overhead_wires",
                                   "district_prefs": [{"district": "C", "strength": "hard"}]}),
        ("text", "Should District C be a hard requirement or a strong preference?"),
        ("tool", "optimize_plan", {"budget_usd": 6000, "n_trees": 6, "exclude": ["overhead_wires"],
                                   "district_prefs": [{"district": "C", "strength": "hard"}]}),
        ("text", "Done."),
    ])
    p = Planner(data(), llm, "fake")
    r1 = p.chat([{"role": "user", "content": ask}])
    res1 = r1["tool_log"][0]["result"]
    assert "needs_clarification" in res1 and p.state.pending_districts == ["C"]
    r2 = p.chat([{"role": "user", "content": ask},
                 {"role": "assistant", "content": r1["reply"]},
                 {"role": "user", "content": "Just a strong preference."}])
    res2 = r2["tool_log"][0]["result"]
    assert "plan_id" in res2, res2
    assert res2["constraints"]["district_prefs"][0]["strength"] == "soft"   # not the LLM's "hard"
    assert "District C: preferred (×2)" in res2["chips"]


def test_explain_excluded_site_and_what_if():
    llm = FakeLLM([
        ("tool", "optimize_plan", {"n_trees": 4, "exclude": ["power_lines"]}),
        ("text", "Planned."),
        ("tool", "explain_site", {"h3_cell": "elm"}),
        ("text", "It is under wires."),
        ("tool", "what_if", {"n_trees": 8}),
        ("text", "More trees."),
    ])
    d = data()
    p = Planner(d, llm, "fake")
    p._sites_by_h3["elm"] = "elm"           # toy site ids aren't H3
    p.chat([{"role": "user", "content": "Plan 4 trees, avoid power lines."}])
    r = p.chat([{"role": "user", "content": "Why not Elm St?"}])
    assert r["tool_log"][0]["result"]["reason"] == "excluded:overhead_wires"
    r = p.chat([{"role": "user", "content": "What if we plant 8?"}])
    res = r["tool_log"][0]["result"]
    assert res["n_trees"] == 8 and res["vs_previous"]["n_trees"]["diff"] == 4


def test_llm_added_constraints_dropped_and_missing_data_reported():
    # Seen live 2026-11-08: the model added "bus_stops" the user never said.
    llm = FakeLLM([
        ("tool", "optimize_plan", {"n_trees": 4, "attractors": ["bus_stops"],
                                   "exclude": ["overhead_wires", "narrow_sidewalk"]}),
        ("text", "ok"),
    ])
    d = data()
    p = Planner(d, llm, "fake")
    r = p.chat([{"role": "user", "content": "Plan 4 trees and stay away from power lines."}])
    res = r["tool_log"][0]["result"]
    assert res["constraints"]["attractors"] == []                    # never mentioned
    assert res["constraints"]["exclude"] == ["overhead_wires"]       # sidewalk never mentioned
    assert "unsupported" not in res                                  # toy data has wire flags
    d.sites["overhead_wires"] = False                                # a city with no wire layer
    p2 = Planner(d, FakeLLM([("tool", "optimize_plan", {"n_trees": 4, "exclude": ["overhead_wires"]}),
                             ("text", "ok")]), "fake")
    res2 = p2.chat([{"role": "user", "content": "4 trees, avoid power lines"}])["tool_log"][0]["result"]
    assert any("overhead_wires" in u for u in res2["unsupported"])


def test_street_name_is_not_a_location():
    llm = FakeLLM([("tool", "optimize_plan", {"n_trees": 2}), ("text", "ok"),
                   ("tool", "explain_site", {"h3_cell": "the cell containing elm street"}),
                   ("text", "Which point on the map?")])
    p = Planner(data(), llm, "fake")
    p.chat([{"role": "user", "content": "plan 2 trees"}])
    r = p.chat([{"role": "user", "content": "why not elm street?"}])
    assert "error" in r["tool_log"][0]["result"]


def test_ungrounded_numbers_flags_invented_figures():
    log = [{"result": {"residents_benefiting": 14000, "mean_cooling_f": 2.21}}]
    assert ungrounded_numbers("Serves 14,000 residents, cools 2.2°F.", log, "") == []
    assert ungrounded_numbers("Serves 22,500 residents.", log, "") == ["22,500"]
