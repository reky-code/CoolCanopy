# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · People: residents and vulnerability per H3 cell
# MAGIC - 2020 Census block groups (TIGERweb): geometry + total population → `bronze_block_groups`
# MAGIC - CDC/ATSDR SVI 2022 (tract): age 65+, age ≤17, poverty <150%, no vehicle, SVI percentile → `bronze_svi_tract`
# MAGIC - `silver_pop_h3`: block-group population spread over its cells in proportion to (impervious % + floor);
# MAGIC   tract shares applied on top. `vulnerable = residents × SVI percentile` (the optimizer's default weight).
# MAGIC
# MAGIC The Census API now requires a key, so this uses keyless sources. Swap in ACS block-group tables
# MAGIC later if a key is added.

# COMMAND ----------

# MAGIC %pip install -q pyyaml requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import io
import os
import sys

import pandas as pd
import requests

nb_dir = os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext()
                         .notebookPath().get())
bundle_root = os.path.abspath(os.path.join("/Workspace" + nb_dir, ".."))
sys.path.insert(0, os.path.join(bundle_root, "src"))
dbutils.widgets.text("city_config", "conf/city_houston.yaml")

from heat.config import load_config
from heat import context

cfg = load_config(os.path.join(bundle_root, dbutils.widgets.get("city_config")))
T, pc = cfg["fq_schema"], cfg["people"]

# COMMAND ----------

bg = pd.DataFrame(context.fetch_arcgis_polygons(
    pc["block_groups_url"], "1=1", "GEOID", extra_fields=["POP100", "INTPTLAT", "INTPTLON"],
    bbox4326=cfg["bbox"]))
bg = bg.rename(columns={"name": "geoid"}).drop(columns=["feature_id"])
bg["pop"] = pd.to_numeric(bg["POP100"], errors="coerce").fillna(0).astype(int)
bg["lat"] = pd.to_numeric(bg["INTPTLAT"], errors="coerce")
bg["lon"] = pd.to_numeric(bg["INTPTLON"], errors="coerce")
# A block group can be split into several polygons; keep pop on the first piece only
bg["pop"] = bg["pop"].where(~bg.duplicated("geoid"), 0)
bg = bg[["geoid", "pop", "lat", "lon", "polygon_geojson"]]
print(f"{bg.geoid.nunique():,} block groups, {bg['pop'].sum():,} people")
spark.createDataFrame(bg).write.mode("overwrite").option("overwriteSchema", "true") \
     .saveAsTable(f"{T}.bronze_block_groups")

# COMMAND ----------

svi = pd.read_csv(io.BytesIO(requests.get(pc["svi_csv_url"], timeout=180).content),
                  dtype={"FIPS": str})
svi = svi[svi["FIPS"].str[:5].isin(pc["counties"])]
keep = ["FIPS", "E_TOTPOP", "E_AGE65", "E_AGE17", "E_POV150", "E_NOVEH", "E_HH", "RPL_THEMES"]
svi = svi[keep].rename(columns={"FIPS": "tract"})
for c in keep[1:]:
    svi[c] = pd.to_numeric(svi[c], errors="coerce")
    svi.loc[svi[c] < 0, c] = None          # SVI uses -999 for missing
print(f"{len(svi):,} tracts in {pc['counties']}")
spark.createDataFrame(svi).write.mode("overwrite").option("overwriteSchema", "true") \
     .saveAsTable(f"{T}.bronze_svi_tract")

# COMMAND ----------

floor = float(pc.get("dasymetric_floor", 5))
spark.sql(f"""
CREATE OR REPLACE TABLE {T}.silver_pop_h3
COMMENT 'Residents and vulnerability per H3 res-10 cell (2020 block-group population, dasymetric by impervious; SVI 2022 tract shares)'
AS
WITH bg_cells AS (           -- cells whose center falls in each block-group polygon
  SELECT geoid, explode(h3_polyfillash3(polygon_geojson, 10)) AS h3_10
  FROM {T}.bronze_block_groups
),
bg_cell AS (SELECT h3_10, min(geoid) AS geoid FROM bg_cells GROUP BY h3_10),
bg_pop AS (SELECT geoid, sum(pop) AS pop, any_value(lat) AS lat, any_value(lon) AS lon
           FROM {T}.bronze_block_groups GROUP BY geoid),
-- tiny block groups that contain no cell center: put them on the cell at their interior point
orphans AS (
  SELECT b.geoid, h3_longlatash3(b.lon, b.lat, 10) AS h3_10
  FROM bg_pop b LEFT ANTI JOIN bg_cell c ON c.geoid = b.geoid
  WHERE b.pop > 0 AND b.lat IS NOT NULL
),
cells AS (SELECT h3_10, geoid FROM bg_cell UNION ALL SELECT h3_10, geoid FROM orphans),
w AS (
  SELECT c.h3_10, c.geoid, p.pop, coalesce(l.imperv_pct_2021, 0) + {floor} AS w
  FROM cells c JOIN bg_pop p USING (geoid)
  LEFT JOIN {T}.silver_landcover_h3 l USING (h3_10)
),
alloc AS (
  SELECT h3_10, geoid, pop * w / sum(w) OVER (PARTITION BY geoid) AS residents FROM w
),
cell AS (   -- a cell can receive an orphan block group on top of its own
  SELECT h3_10, max_by(geoid, residents) AS geoid, sum(residents) AS residents
  FROM alloc GROUP BY h3_10
)
SELECT c.h3_10, c.geoid AS block_group, substr(c.geoid, 1, 11) AS tract,
       c.residents,
       c.residents * s.E_AGE65  / nullif(s.E_TOTPOP, 0) AS age65,
       c.residents * s.E_AGE17  / nullif(s.E_TOTPOP, 0) AS age17,
       c.residents * s.E_POV150 / nullif(s.E_TOTPOP, 0) AS pov150,
       s.E_NOVEH / nullif(s.E_HH, 0)                       AS noveh_hh_share,
       s.RPL_THEMES                                        AS svi_rank,
       c.residents * coalesce(s.RPL_THEMES, 0.5)           AS vulnerable
FROM cell c LEFT JOIN {T}.bronze_svi_tract s ON s.tract = substr(c.geoid, 1, 11)
""")

display(spark.sql(f"""
  SELECT round(sum(p.residents)) AS residents_in_city, round(sum(p.age65)) AS age65,
         round(sum(p.vulnerable)) AS vulnerable, round(avg(p.svi_rank), 3) AS mean_svi
  FROM {T}.silver_pop_h3 p JOIN {T}.silver_city_h3 c USING (h3_10)"""))
