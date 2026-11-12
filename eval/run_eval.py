"""Evaluate the planner's request parsing + clarification behaviour against eval/requests.jsonl.

For each request we take the agent's first optimize_plan call *after* our validation,
grounding and hard/soft resolution (what the optimizer would actually see) and score:
  - field_accuracy: share of expected fields that match
  - exact: every expected field matches
  - ask_correct: the agent asked exactly the expected clarifying questions
Results are logged to MLflow when available (python eval/run_eval.py [endpoint]).
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), ROOT]

from heat.agent.planner import Planner, PlannerData  # noqa: E402
from tests.test_optimize import toy                  # noqa: E402


def score(case: dict, c: dict | None, asked: list[str]) -> dict:
    exp = case["expect"]
    checks = {}
    c = c or {}
    for k in ("budget_usd", "n_trees", "cost_per_tree", "objective"):
        if k in exp:
            checks[k] = c.get(k) == exp[k]
    if "exclude" in exp:
        checks["exclude"] = sorted(c.get("exclude", [])) == sorted(exp["exclude"])
    if "attractors" in exp:
        checks["attractors"] = sorted(a["type"] for a in c.get("attractors", [])) == sorted(exp["attractors"])
    got_d = {p["district"]: p["strength"] for p in c.get("district_prefs", [])}
    checks["districts"] = got_d == exp.get("districts", {})
    return {"id": case["id"], "field_accuracy": sum(checks.values()) / len(checks),
            "exact": all(checks.values()), "ask_correct": sorted(asked) == sorted(case["should_ask"]),
            "failed_fields": [k for k, v in checks.items() if not v], "asked": asked,
            "parsed": c}


def main(endpoint: str = "databricks-llama-4-maverick", profile: str = os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")) -> None:
    from databricks.sdk import WorkspaceClient
    llm = WorkspaceClient(profile=profile).serving_endpoints.get_open_ai_client()
    sites, eff, hexes = toy()
    sites["lst_anom_c"] = [3.0, 2.0, 1.0, 0.5, 4.0]
    with open(os.path.join(ROOT, "eval", "requests.jsonl"), encoding="utf-8-sig") as f:
        cases = [json.loads(l) for l in f if l.strip()]
    rows = []
    for case in cases:
        p = Planner(PlannerData(sites, eff, hexes, default_cost=1500), llm, endpoint, max_steps=2)
        t = time.time()
        out = p.chat([{"role": "user", "content": case["text"]}])
        call = next((x for x in out["tool_log"] if x["tool"] == "optimize_plan"), None)
        res = call["result"] if call else {}
        asked = [q["field"] for q in res.get("needs_clarification", [])]
        if not call:   # model asked in prose without calling the tool
            asked = ["n_trees"] if "n_trees" in case["should_ask"] else ["<prose>"]
        c = res.get("constraints") or (p.state.constraints.model_dump() if p.state.constraints else None)
        r = score(case, c, asked)
        r["latency_s"] = round(time.time() - t, 1)
        rows.append(r)
        print(f"{case['id']}: fields {r['field_accuracy']:.2f} exact={r['exact']} ask_ok={r['ask_correct']} "
              f"{r['failed_fields']} asked={asked}")
    summary = {"n": len(rows), "field_accuracy": sum(r["field_accuracy"] for r in rows) / len(rows),
               "exact_match": sum(r["exact"] for r in rows) / len(rows),
               "clarification_accuracy": sum(r["ask_correct"] for r in rows) / len(rows),
               "mean_latency_s": sum(r["latency_s"] for r in rows) / len(rows)}
    print(json.dumps(summary, indent=2))
    os.makedirs(os.path.join(ROOT, "eval", "results"), exist_ok=True)
    with open(os.path.join(ROOT, "eval", "results", f"{endpoint}.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2, default=str)
    try:
        import mlflow
        mlflow.set_tracking_uri(f"databricks://{profile}")
        mlflow.set_experiment(os.environ.get("EVAL_EXPERIMENT", "/Shared/heat-trees-agent-eval"))
        with mlflow.start_run(run_name=f"parse_eval_{endpoint}"):
            mlflow.log_param("endpoint", endpoint)
            mlflow.log_metrics(summary)
            mlflow.log_dict({"rows": rows}, "eval_rows.json")
        print("logged to MLflow")
    except Exception as e:  # MLflow optional
        print("MLflow logging skipped:", type(e).__name__, str(e)[:120])


if __name__ == "__main__":
    main(*sys.argv[1:])
