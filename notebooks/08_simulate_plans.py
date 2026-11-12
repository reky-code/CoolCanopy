# Databricks notebook source
# MAGIC %md
# MAGIC # 08 · Simulate planting strategies
# MAGIC Five strategies, each scored with the same effect table and diminishing-returns rule:
# MAGIC `random`, `even` (equal per council district), `hottest_first`, `cooling` (greedy on Δ°C × residents),
# MAGIC `vulnerability` (greedy on Δ°C × SVI-weighted residents, the default).
# MAGIC
# MAGIC Greedy picks are nested (first n of the best N = best n), so one run up to the largest size gives
# MAGIC every slider frame. Outputs: `gold_plans`, `gold_plan_picks`, `gold_plan_hex_delta`, `gold_decision_log`.

# COMMAND ----------

# MAGIC %pip install -q pydantic pyyaml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import sys
import time

import numpy as np
import pandas as pd

nb_dir = os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext()
                         .notebookPath().get())
bundle_root = os.path.abspath(os.path.join("/Workspace" + nb_dir, ".."))
sys.path.insert(0, os.path.join(bundle_root, "src"))
dbutils.widgets.text("city_config", "conf/city_houston.yaml")

from heat.config import load_config
from heat.agent.constraints import PlanConstraints
from heat.sim.optimize import baseline_order, optimize, simulate_sequence

cfg = load_config(os.path.join(bundle_root, dbutils.widgets.get("city_config")))
T = cfg["fq_schema"]
pl = cfg["planting"]
sizes = sorted(pl["plan_sizes"]); N = sizes[-1]
sat = float(pl["saturation_c"]); cost = float(cfg["agent"]["default_cost_per_tree"])

sites = spark.table(f"{T}.gold_planting_sites").toPandas()
sites = sites.rename(columns={"site_hex": "hex"}).assign(site_id=lambda d: d["hex"])
effects = spark.sql(f"""SELECT site_hex AS site_id, target_hex, d_temp_c, d_temp_lo, d_temp_hi
                        FROM {T}.gold_planting_effects""").toPandas()
hexes = spark.sql(f"""
  SELECT g.h3_10 AS hex, coalesce(p.residents, 0) AS residents,
         coalesce(p.vulnerable, 0) AS vulnerable, g.lst_anom_c, g.district
  FROM {T}.gold_hex_features g LEFT JOIN {T}.silver_pop_h3 p USING (h3_10)""").toPandas()
hot_cut = hexes.loc[hexes.residents > 0, "lst_anom_c"].quantile(0.9)
hot_hexes = set(hexes.loc[(hexes.lst_anom_c >= hot_cut) & (hexes.residents > 0), "hex"])
print(f"{len(sites):,} sites, {len(effects):,} effect rows, {len(hexes):,} cells; hottest-decile cut {hot_cut:.2f} C")

# COMMAND ----------

results = {}
for strat in ("cooling", "vulnerability"):
    t = time.time()
    r = optimize(sites, effects, hexes, PlanConstraints(n_trees=N, objective=strat),
                 default_cost=cost, saturation_c=sat)
    # replay to get the slider frames (identical result, see test_simulate_sequence_matches_optimizer)
    replay = simulate_sequence(sites, effects, hexes, r.picks.site_id.tolist(), sat, cost, sizes)
    replay.decision_log = r.decision_log
    results[strat] = replay
    print(strat, f"{time.time()-t:.0f}s", r.summary)
for strat in ("random", "even", "hottest_first"):
    t = time.time()
    results[strat] = simulate_sequence(sites, effects, hexes, baseline_order(sites, N, strat),
                                       sat, cost, sizes)
    print(strat, f"{time.time()-t:.0f}s", results[strat].summary)

# COMMAND ----------

plan_rows, pick_rows, delta_rows = [], [], []
for strat, r in results.items():
    for n in sizes:
        hd, s = r.frames.get(n, (r.hex_delta, r.summary))
        hot_reached = int(hd[(hd.d_temp_c <= -0.05) & hd.hex.isin(hot_hexes)].shape[0])
        plan_rows.append({
            "plan_id": f"{strat}_{n}", "strategy": strat, "n_trees": n,
            "cost_usd": n * cost,
            "residents_benefiting": s["residents_benefiting"],
            "vulnerable_benefiting": s["vulnerable_benefiting"],
            "degree_person_c": s["degree_person_c"],
            "degree_vulnerable_c": s["degree_vulnerable_c"],
            "mean_cooling_benefiting_hexes_c": s["mean_cooling_benefiting_hexes_c"],
            "mean_cooling_lo_c": (s.get("mean_cooling_range_c") or [None, None])[0],
            "mean_cooling_hi_c": (s.get("mean_cooling_range_c") or [None, None])[1],
            "hot_decile_hexes_reached": hot_reached,
            "trees_by_district_json": json.dumps(
                r.picks.head(n)["district"].fillna("none").value_counts().to_dict()),
        })
        delta_rows.append(hd.assign(plan_id=f"{strat}_{n}"))
    pick_rows.append(r.picks.assign(strategy=strat)[["strategy", "order", "site_id", "district"]])

plans = pd.DataFrame(plan_rows)
spark.createDataFrame(plans).write.mode("overwrite").option("overwriteSchema", "true") \
     .saveAsTable(f"{T}.gold_plans")
spark.createDataFrame(pd.concat(pick_rows).rename(columns={"site_id": "site_hex"})) \
     .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{T}.gold_plan_picks")
spark.createDataFrame(pd.concat(delta_rows).rename(columns={"hex": "h3_10"})) \
     .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{T}.gold_plan_hex_delta")

log = results["vulnerability"].decision_log.rename(columns={"site_id": "site_hex"})
log["plan_strategy"] = "vulnerability"
spark.createDataFrame(log.drop(columns=[c for c in ("segment_name",) if c in log])) \
     .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{T}.gold_decision_log")

display(spark.sql(f"""
  SELECT strategy, n_trees, residents_benefiting, vulnerable_benefiting,
         round(degree_vulnerable_c) AS degree_vulnerable_c, hot_decile_hexes_reached
  FROM {T}.gold_plans WHERE n_trees = {N} ORDER BY degree_vulnerable_c DESC"""))
