from heat.agent.constraints import PlanConstraints, clarifications, reconcile, strength_stated

MANAGER = ("We've got $250k for about 600 trees this season. Stay away from power lines, "
           "prioritize District C, and favor bus stops.")


def test_coerces_qwen_style_stringified_args():
    # Exact shape returned by databricks-qwen35-122b-a10b in the 2026-11-07 test
    raw = {"n_trees": "600", "budget_usd": "250000",
           "exclude": "[\"overhead_wires\", \"transmission\"]",
           "district_prefs": "[{\"district\": 3, \"strength\": \"hard\"}]",
           "attractors": "[\"bus_stops\"]"}
    c = PlanConstraints.model_validate(raw)
    assert c.n_trees == 600 and c.budget_usd == 250000
    assert c.exclude == ["overhead_wires", "transmission"]
    assert c.district_prefs[0].district == "3"  # int coerced to str; numbered cities still work
    assert c.attractors[0].type == "bus_stops"


def test_district_letter_normalization():
    c = PlanConstraints(district_prefs=[{"district": "district c"}])
    assert c.district_prefs[0].district == "C"


def test_coerces_gpt_oss_scalar_exclude():
    c = PlanConstraints.model_validate({"n_trees": 600, "exclude": "overhead_wires"})
    assert c.exclude == ["overhead_wires"]


def test_power_lines_alias():
    assert PlanConstraints(exclude=["power_lines"]).exclude == ["overhead_wires"]


def test_prioritize_is_ambiguous_and_triggers_question():
    c = PlanConstraints.model_validate(
        {"n_trees": 600, "budget_usd": 250000,
         "district_prefs": [{"district": "C", "strength": "hard"}]})  # LLM guessed "hard"
    c = reconcile(c, MANAGER)
    assert c.district_prefs[0].strength is None
    fields = [q.field for q in clarifications(c, default_cost=1500)]
    assert "district_prefs[C].strength" in fields
    assert "cost_per_tree" in fields  # $417/tree vs $1,500 table


def test_explicit_quota_is_hard():
    assert strength_stated("at least 40% in District C") == "hard"
    assert strength_stated("ideally lean toward District C") == "soft"
    assert strength_stated("prioritize District C") is None
    # answers to the clarifying question
    assert strength_stated("Just a strong preference.") == "soft"
    assert strength_stated("Hard requirement, at least 40%") == "hard"


def test_resolved_trees_capped_by_budget():
    c = PlanConstraints(budget_usd=250_000, n_trees=600)
    assert c.resolved_n_trees(default_cost=1500) == 166
    c.cost_per_tree = 400
    assert c.resolved_n_trees(default_cost=1500) == 600


def test_chips():
    c = PlanConstraints(budget_usd=250_000, n_trees=600, exclude=["overhead_wires"],
                        district_prefs=[{"district": "C"}], attractors=["bus_stops"])
    chips = c.chips()
    assert "Budget $250,000" in chips and "District C: priority (hard or soft?)" in chips

