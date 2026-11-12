"""Super Neighborhoods as a second planning unit next to council districts."""
from heat.agent.constraints import PlanConstraints, area_label, area_pattern, is_district_code, strength_stated
from heat.agent.planner import Planner
from heat.sim.optimize import in_area, optimize
from tests.test_optimize import toy
from tests.test_planner import FakeLLM, data


def nb_sites():
    sites, eff, hexes = toy()
    sites["super_nbhd"] = ["GULFTON", "GULFTON", "GREATER UPTOWN", "GREATER UPTOWN", "GULFTON"]
    return sites, eff, hexes


def test_codes_vs_names():
    assert is_district_code("J") and is_district_code("3") and not is_district_code("GULFTON")
    assert area_label("J") == "District J" and area_label("GREATER UPTOWN") == "Greater Uptown"
    assert area_pattern("GREATER UPTOWN").search("plant near uptown please")
    assert not area_pattern("GREATER UPTOWN").search("greater good")
    assert strength_stated("at least 30% in Gulfton, and favor bus stops", "GULFTON") == "hard"
    assert strength_stated("avoid wires and prefer Gulfton and Sharpstown.", "SHARPSTOWN") == "soft"
    assert strength_stated("favor bus stops and prioritize District 3", "3") is None


def test_in_area_matches_district_or_neighborhood():
    sites, *_ = nb_sites()
    assert in_area(sites, "c").tolist() == [False, False, True, True, False]
    assert in_area(sites, "Greater Uptown").sum() == 2
    assert not in_area(sites.drop(columns="super_nbhd"), "GULFTON").any()


def test_hard_neighborhood_quota():
    sites, eff, hexes = nb_sites()
    c = PlanConstraints(n_trees=6, district_prefs=[
        {"district": "Greater Uptown", "strength": "hard", "min_share": 0.5}])
    r = optimize(sites, eff, hexes, c, default_cost=1000)
    up = set(sites.site_id[in_area(sites, "GREATER UPTOWN")])
    assert r.picks.site_id.isin(up).sum() >= 3


def test_planner_grounds_and_resolves_neighborhood_name():
    d = data()
    d.sites["super_nbhd"] = ["GULFTON", "GULFTON", "GREATER UPTOWN", "GREATER UPTOWN", "GULFTON"]
    ask = "Plan 6 trees and prefer Uptown. Ignore District K."
    llm = FakeLLM([
        ("tool", "optimize_plan", {"n_trees": 6, "district_prefs": [
            {"district": "uptown", "strength": "soft"}, {"district": "Montrose", "strength": "soft"}]}),
        ("text", "Done."),
    ])
    p = Planner(d, llm, "fake")
    assert p.resolve_neighborhood("2nd uptown") == "GREATER UPTOWN"
    assert p.resolve_neighborhood("Gulfon") == "GULFTON"          # typo -> close match
    assert p.resolve_neighborhood("Montrose") is None
    res = p.chat([{"role": "user", "content": ask}])["tool_log"][0]["result"]
    prefs = res["constraints"]["district_prefs"]
    assert [x["district"] for x in prefs] == ["GREATER UPTOWN"]   # Montrose was never said
    assert "Greater Uptown: preferred (×2)" in res["chips"]
    assert "Greater Uptown" in res["trees_in_named_areas"]
