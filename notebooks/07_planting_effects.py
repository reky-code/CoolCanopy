# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Planting candidates + effect table
# MAGIC **Candidates** (one site per H3 cell for now): inside city limits, < 20% water, existing canopy < 40%,
# MAGIC impervious 10–95%, and room for at least one tree. **Capacity** counts tree spots on open ground
# MAGIC (off buildings + setback and off roads, from `04b`), times a usable share; without footprints it
# MAGIC falls back to NLCD open, unshaded ground. `gold_planting_spots` has the exact spot of each tree.
# MAGIC
# MAGIC **Effects**: for each site and each target within 2 rings, the modelled temperature change from ONE
# MAGIC young tree (~50 m² crown at ~10 years), from the point model plus the 5 spatial-fold models
# MAGIC (10th–90th percentile → uncertainty). Output feeds the optimizer.

# COMMAND ----------

# MAGIC %pip install -q lightgbm pyyaml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import glob
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

nb_dir = os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext()
                         .notebookPath().get())
bundle_root = os.path.abspath(os.path.join("/Workspace" + nb_dir, ".."))
sys.path.insert(0, os.path.join(bundle_root, "src"))
dbutils.widgets.text("city_config", "conf/city_houston.yaml")

from heat.config import load_config
from heat import model as hm

cfg = load_config(os.path.join(bundle_root, dbutils.widgets.get("city_config")))
T, city = cfg["fq_schema"], cfg["city"]
pl = cfg.get("planting", {})
crown_m2 = float(pl.get("crown_m2", 50))
art = f"/Volumes/{cfg['catalog']}/{cfg['schema']}/artifacts/{city}/models"
models = [lgb.Booster(model_file=f"{art}/full.txt")] + \
         [lgb.Booster(model_file=p) for p in sorted(glob.glob(f"{art}/fold_*.txt"))]
print(len(models), "models")

# COMMAND ----------

# Capacity = trees that fit on open ground.
#  - footprints (04b): open tree spots off buildings/roads x usable share
#  - fallback (no footprint table, or a cell it missed): NLCD open, unshaded m2 x usable share / spot m2
# Cells with no room for even one tree are dropped.
usable = float(pl.get("usable_share", 0.5))
max_trees = int(pl.get("max_trees_per_cell", 30))
spot_m2 = float(pl.get("fallback_spot_m2", 300))
has_fp = spark.catalog.tableExists(f"{T}.silver_plantable_h3") and \
         spark.catalog.tableExists(f"{T}.silver_plantable_spots")
print("footprint-based capacity:", has_fp)

fallback = f"""floor(15047.5 * greatest(0, 1 - g.imperv_pct_2021 / 100 - g.canopy_pct_2021 / 100)
                     * {usable} / {spot_m2})"""
fp_join = f"""
LEFT JOIN {T}.silver_plantable_h3 pa USING (h3_10)
LEFT JOIN (SELECT h3_10, count(*) AS n_spots FROM {T}.silver_plantable_spots GROUP BY 1) sp USING (h3_10)
""" if has_fp else ""
fp_cols = ("pa.open_frac AS open_frac, pa.n_buildings AS n_buildings, coalesce(sp.n_spots, 0) AS n_spots,"
           if has_fp else "CAST(NULL AS DOUBLE) AS open_frac, CAST(NULL AS BIGINT) AS n_buildings, "
                          "CAST(NULL AS BIGINT) AS n_spots,")
cap_expr = (f"CASE WHEN pa.h3_10 IS NOT NULL THEN floor(coalesce(sp.n_spots, 0) * {usable}) ELSE {fallback} END"
            if has_fp else fallback)

spark.sql(f"""
CREATE OR REPLACE TABLE {T}.gold_planting_sites
COMMENT 'Candidate planting sites (one per H3 res-10 cell). capacity = open tree spots off buildings and roads (Overture footprints, 04b) x {usable} usable share, max {max_trees}; capacity_source says whether the NLCD open-ground fallback was used'
AS
WITH c AS (
  SELECT g.h3_10, g.district, g.super_nbhd, g.canopy_pct_2021, g.imperv_pct_2021, g.lst_anom_c,
         g.center_wkt, p.residents, p.vulnerable, p.age65, p.svi_rank, {fp_cols}
         CAST(least({max_trees}, {cap_expr}) AS INT) AS capacity,
         {"CASE WHEN pa.h3_10 IS NOT NULL THEN 'footprints' ELSE 'nlcd_open_ground' END"
          if has_fp else "'nlcd_open_ground'"} AS capacity_source
  FROM {T}.gold_hex_features g
  LEFT JOIN {T}.silver_pop_h3 p USING (h3_10)
  {fp_join}
  WHERE g.water_frac < 0.2 AND g.canopy_pct_2021 < 40 AND g.imperv_pct_2021 BETWEEN 10 AND 95
)
SELECT h3_10 AS site_hex, h3_h3tostring(h3_10) AS site_id, district, super_nbhd,
       capacity, capacity_source, open_frac, n_buildings, n_spots,
       canopy_pct_2021, imperv_pct_2021, lst_anom_c,
       coalesce(residents, 0) AS residents, coalesce(vulnerable, 0) AS vulnerable,
       coalesce(age65, 0) AS age65, svi_rank,
       FALSE AS overhead_wires, FALSE AS transmission, FALSE AS narrow_sidewalk,
       center_wkt
FROM c WHERE capacity >= 1
""")

if has_fp:   # exact tree locations: the first `capacity` spots of each site (stable random order)
    spark.sql(f"""
    CREATE OR REPLACE TABLE {T}.gold_planting_spots
    COMMENT 'Tree spots for each planting site, off buildings (+setback) and roads: a plan with k trees at a site uses spot_rank 1..k'
    AS SELECT s.site_hex, h3_h3tostring(p.spot) AS spot, p.spot_rank, p.lon, p.lat
    FROM {T}.gold_planting_sites s
    JOIN {T}.silver_plantable_spots p ON p.h3_10 = s.site_hex AND p.spot_rank <= s.capacity
    """)

display(spark.sql(f"""
  SELECT capacity_source, count(*) AS sites, sum(capacity) AS tree_slots, round(avg(capacity), 1) AS avg_cap,
         round(avg(open_frac), 2) AS open_frac
  FROM {T}.gold_planting_sites GROUP BY 1"""))

pairs = spark.sql(f"""
  SELECT s.site_hex, t AS target_hex, h3_distance(s.site_hex, t) AS k
  FROM {T}.gold_planting_sites s LATERAL VIEW explode(h3_kring(s.site_hex, 2)) x AS t
""").toPandas()

feat = spark.sql(f"""
  SELECT h3_10, canopy_pct_2021, canopy_k1, canopy_k2, imperv_pct_2021, imperv_k1, imperv_k2,
         water_frac,
         CAST(regexp_extract(center_wkt, 'POINT\\\\(([-0-9.]+) ', 1) AS DOUBLE) AS lon,
         CAST(regexp_extract(center_wkt, ' ([-0-9.]+)\\\\)', 1) AS DOUBLE) AS lat
  FROM {T}.gold_hex_features
""").toPandas().dropna()
print(f"{pairs.site_hex.nunique():,} sites, {len(pairs):,} site-target pairs, {len(feat):,} cells")

# COMMAND ----------

eff = hm.planting_effects(feat, pairs, models, crown_m2=crown_m2)
print(eff.describe().T[["mean", "min", "50%", "max"]])
spark.createDataFrame(eff).write.mode("overwrite").option("overwriteSchema", "true") \
     .saveAsTable(f"{T}.gold_planting_effects")

display(spark.sql(f"""
  SELECT k, count(*) AS n, round(avg(d_temp_c), 4) AS mean_dT_per_tree_c,
         round(avg(d_temp_lo), 4) AS lo, round(avg(d_temp_hi), 4) AS hi
  FROM {T}.gold_planting_effects GROUP BY k ORDER BY k"""))
