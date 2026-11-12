"""Create the 'Houston Heat & Trees' Genie space over the gold tables, or update it in place when
eval/results/genie_space.json already names one (keeps the space id and URL).

usage: python scripts/create_genie_space.py [warehouse_id] [profile] [--new]
  warehouse_id defaults to $DATABRICKS_WAREHOUSE_ID; profile to $DATABRICKS_CONFIG_PROFILE (else DEFAULT)
"""
import json
import os
import subprocess
import sys
import uuid

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
WAREHOUSE = _args[0] if _args else os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
PROFILE = _args[1] if len(_args) > 1 else os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
if not WAREHOUSE:
    sys.exit("Pass a SQL warehouse id or set DATABRICKS_WAREHOUSE_ID.")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def i() -> str:
    return uuid.uuid4().hex


space = {
    "version": 2,
    "config": {"sample_questions": [
        {"id": i(), "question": ["Which council districts are hottest, and how much canopy do they have?"]},
        {"id": i(), "question": ["Which neighborhoods with lots of elderly residents are hottest and have the least canopy?"]},
        {"id": i(), "question": ["How much did tree canopy change from 2011 to 2021 by district?"]},
        {"id": i(), "question": ["Compare planting strategies for 1,000 trees"]},
        {"id": i(), "question": ["Which Super Neighborhoods have the most residents in hot cells?"]},
    ]},
    "data_sources": {"tables": [
        {"identifier": "workspace.heat.genie_cells"},
        {"identifier": "workspace.heat.gold_plans"},
    ]},
    "instructions": {
        "text_instructions": [{"id": i(), "content": [
            "* Data covers Houston, TX city limits on an H3 res-10 grid (cells of ~15,000 m2).\n",
            "* surface_heat_vs_city_c/_f is the SUMMER SURFACE temperature anomaly vs the city median "
            "(Landsat 8/9, ~10:30am, 2019-2024), not air temperature. Always say 'surface'.\n",
            "* Prefer Fahrenheit (_f columns) in answers for Houston users.\n",
            "* 'Elderly' means residents_65_plus. 'Vulnerable' means social_vulnerability_percentile (CDC SVI 2022).\n",
            "* When aggregating by district, weight heat by residents when the question is about people.\n",
            "* council_district is a letter A-K (council map approved 10-12-2022). NULL = edge cell outside the polygons.\n",
            "* super_neighborhood is one of the city's 88 Super Neighborhoods (resident-led planning areas, e.g. "
            "Gulfton, Sharpstown, Greater Uptown). When users name a neighborhood, filter with "
            "super_neighborhood ILIKE '%name%'. NULL = edge cell outside the polygons.\n",
            "* gold_plans holds precomputed planting strategies; plan_id = strategy_ntrees. "
            "vulnerable_benefiting counts SVI-weighted residents in cells cooled by at least 0.05 C.\n",
        ]}],
        "example_question_sqls": [
            {"id": i(), "question": ["Hottest districts by population-weighted surface heat"],
             "sql": ["SELECT council_district,\n",
                     "  round(sum(surface_heat_vs_city_f * residents) / sum(residents), 2) AS pop_weighted_heat_f,\n",
                     "  round(avg(tree_canopy_pct), 1) AS mean_canopy_pct, sum(residents) AS residents\n",
                     "FROM workspace.heat.genie_cells\n",
                     "WHERE council_district IS NOT NULL AND residents > 0\n",
                     "GROUP BY council_district ORDER BY pop_weighted_heat_f DESC"]},
            {"id": i(), "question": ["Hottest Super Neighborhoods by population-weighted surface heat"],
             "sql": ["SELECT super_neighborhood,\n",
                     "  round(sum(surface_heat_vs_city_f * residents) / sum(residents), 2) AS pop_weighted_heat_f,\n",
                     "  round(avg(tree_canopy_pct), 1) AS mean_canopy_pct, sum(residents) AS residents\n",
                     "FROM workspace.heat.genie_cells\n",
                     "WHERE super_neighborhood IS NOT NULL AND residents > 0\n",
                     "GROUP BY super_neighborhood HAVING sum(residents) >= 2000\n",
                     "ORDER BY pop_weighted_heat_f DESC LIMIT 15"]},
        ],
    },
}
# The API expects each list sorted by id
space["config"]["sample_questions"].sort(key=lambda q: q["id"])
space["instructions"]["text_instructions"].sort(key=lambda q: q["id"])
space["instructions"]["example_question_sqls"].sort(key=lambda q: q["id"])

body = {"warehouse_id": WAREHOUSE, "serialized_space": json.dumps(space),
        "title": "Houston Heat & Trees",
        "description": "Ask about summer surface heat, tree canopy, residents and planting plans in Houston."}
path = os.path.join(ROOT, "eval", "results", "genie_body.json")
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(body, f)
id_path = os.path.join(ROOT, "eval", "results", "genie_space.json")
existing = None
if os.path.exists(id_path) and "--new" not in sys.argv:   # update in place: same id and URL
    existing = json.load(open(id_path))["space_id"]
cmd = ["databricks", "genie", "update-space", existing] if existing else ["databricks", "genie", "create-space"]
out = subprocess.run(cmd + ["--json", f"@{path}", "-o", "json", "-p", PROFILE],
                     capture_output=True, text=True, shell=os.name == "nt")
print(out.stdout[-2000:], out.stderr[-2000:])
if out.returncode == 0:
    sid = json.loads(out.stdout or "{}").get("space_id", existing)
    with open(id_path, "w") as f:
        json.dump({"space_id": sid}, f)
    print("space_id", sid)
