# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · Export app data (Parquet in the artifacts volume)
# MAGIC The Databricks App runs the optimizer in-process, so it needs the sites, effects and
# MAGIC cell weights quickly at startup. Map layers are pre-aggregated: citywide at H3 res 9, plan
# MAGIC deltas at res 10.

# COMMAND ----------

import os

T = "workspace.heat"
out = "/Volumes/workspace/heat/artifacts/houston/app"
os.makedirs(out, exist_ok=True)

queries = {
    "sites": f"""
        SELECT site_hex AS site_id, site_hex AS hex, site_id AS h3_cell, district, super_nbhd, capacity,
               lst_anom_c, canopy_pct_2021, imperv_pct_2021, residents, vulnerable, age65, svi_rank,
               overhead_wires, transmission, narrow_sidewalk
        FROM {T}.gold_planting_sites""",
    "effects": f"""SELECT site_hex AS site_id, target_hex, CAST(d_temp_c AS FLOAT) d_temp_c,
                          CAST(d_temp_lo AS FLOAT) d_temp_lo, CAST(d_temp_hi AS FLOAT) d_temp_hi
                   FROM {T}.gold_planting_effects""",
    "hexes": f"""
        SELECT g.h3_10 AS hex, coalesce(p.residents, 0) AS residents, coalesce(p.vulnerable, 0) AS vulnerable,
               coalesce(p.age65, 0) AS age65, g.lst_anom_c, g.district, g.super_nbhd,
               g.canopy_pct_2021, g.imperv_pct_2021
        FROM {T}.gold_hex_features g LEFT JOIN {T}.silver_pop_h3 p USING (h3_10)""",
    "map_r9": f"""
        SELECT *, CAST(regexp_extract(h3_centeraswkt(h3), ' ([-0-9.]+)\\\\)', 1) AS DOUBLE) AS lat,
                  CAST(regexp_extract(h3_centeraswkt(h3), 'POINT\\\\(([-0-9.]+) ', 1) AS DOUBLE) AS lon
        FROM (
        SELECT h3_h3tostring(h3_toparent(g.h3_10, 9)) AS h3, avg(g.lst_anom_c) AS lst_anom_c,
               avg(g.canopy_pct_2021) AS canopy_pct, avg(g.canopy_pct_2011) AS canopy_2011,
               avg(g.imperv_pct_2021) AS imperv_pct,
               sum(coalesce(p.residents, 0)) AS residents, sum(coalesce(p.age65, 0)) AS age65,
               sum(coalesce(p.vulnerable, 0)) AS vulnerable,
               try_divide(sum(p.svi_rank * p.residents), sum(p.residents)) AS svi,
               max_by(g.district, coalesce(p.residents, 0)) AS district,
               max_by(g.super_nbhd, coalesce(p.residents, 0)) AS super_nbhd
        FROM {T}.gold_hex_features g LEFT JOIN {T}.silver_pop_h3 p USING (h3_10)
        GROUP BY 1)""",
    # one row per Super Neighborhood, same measures as the district rollup + its main district
    "neighborhoods": f"""
        SELECT g.super_nbhd, count(*) AS cells,
               max_by(g.district, coalesce(p.residents, 0)) AS district,
               try_divide(sum(g.lst_anom_c * p.residents), sum(p.residents)) AS heat_pop_c,
               avg(g.lst_anom_c) AS heat_area_c,
               avg(g.canopy_pct_2011) AS canopy_2011, avg(g.canopy_pct_2021) AS canopy_2021,
               avg(g.imperv_pct_2021) AS imperv,
               sum(coalesce(p.residents, 0)) AS residents, sum(coalesce(p.age65, 0)) AS age65,
               sum(coalesce(p.vulnerable, 0)) AS vulnerable,
               sum(CASE WHEN g.lst_anom_c > 3 THEN coalesce(p.residents, 0) ELSE 0 END) AS residents_hot,
               count(s.site_hex) AS candidate_cells
        FROM {T}.gold_hex_features g
        LEFT JOIN {T}.silver_pop_h3 p USING (h3_10)
        LEFT JOIN {T}.gold_planting_sites s ON s.site_hex = g.h3_10
        WHERE g.super_nbhd IS NOT NULL
        GROUP BY g.super_nbhd""",
    # one row per council district (resident-weighted heat; canopy trend 2011 -> 2021)
    "districts": f"""
        SELECT g.district, count(*) AS cells,
               try_divide(sum(g.lst_anom_c * p.residents), sum(p.residents)) AS heat_pop_c,
               avg(g.lst_anom_c) AS heat_area_c,
               avg(g.canopy_pct_2011) AS canopy_2011, avg(g.canopy_pct_2016) AS canopy_2016,
               avg(g.canopy_pct_2021) AS canopy_2021, avg(g.imperv_pct_2021) AS imperv,
               sum(coalesce(p.residents, 0)) AS residents, sum(coalesce(p.age65, 0)) AS age65,
               sum(coalesce(p.vulnerable, 0)) AS vulnerable,
               sum(CASE WHEN g.lst_anom_c > 3 THEN coalesce(p.residents, 0) ELSE 0 END) AS residents_hot,
               count(s.site_hex) AS candidate_cells
        FROM {T}.gold_hex_features g
        LEFT JOIN {T}.silver_pop_h3 p USING (h3_10)
        LEFT JOIN {T}.gold_planting_sites s ON s.site_hex = g.h3_10
        WHERE g.district IS NOT NULL
        GROUP BY g.district""",
    # observed: mean anomaly by canopy bin within pavement bands (the "trees cool" chart)
    "canopy_curve": f"""
        SELECT CAST(floor(least(canopy_pct_2021, 69.99) / 5) * 5 AS INT) AS canopy_bin,
               CASE WHEN imperv_pct_2021 < 30 THEN 'Low pavement (<30%)'
                    WHEN imperv_pct_2021 < 60 THEN 'Medium pavement (30–60%)'
                    ELSE 'High pavement (≥60%)' END AS imperv_band,
               avg(lst_anom_c) AS anom_c, count(*) AS n
        FROM {T}.gold_hex_features WHERE water_frac < 0.1
        GROUP BY 1, 2 HAVING count(*) >= 30""",
    "importance": f"SELECT * FROM {T}.gold_model_importance",
    "dependence": f"SELECT * FROM {T}.gold_model_canopy_dependence",
    "oof": f"""
        SELECT o.pred_oof_c, g.lst_anom_c, g.district, g.super_nbhd
        FROM {T}.gold_model_oof o JOIN {T}.gold_hex_features g USING (h3_10)
        ORDER BY hash(o.h3_10) LIMIT 5000""",
    "plans": f"SELECT * FROM {T}.gold_plans",
    "frames": f"""SELECT plan_id, h3_h3tostring(h3_10) AS h3, CAST(d_temp_c AS FLOAT) d_temp_c
                  FROM {T}.gold_plan_hex_delta""",
    "picks": f"""SELECT strategy, `order`, h3_h3tostring(site_hex) AS h3, district
                 FROM {T}.gold_plan_picks""",
    "metrics": f"SELECT * FROM {T}.gold_model_metrics",
}
if spark.catalog.tableExists(f"{T}.gold_planting_spots"):   # one row per tree spot (off buildings/roads)
    queries["spots"] = f"""SELECT h3_h3tostring(site_hex) AS h3, CAST(spot_rank AS SMALLINT) AS spot_rank,
                                  CAST(lon AS DOUBLE) AS lon, CAST(lat AS DOUBLE) AS lat
                           FROM {T}.gold_planting_spots"""
for name, q in queries.items():
    pdf = spark.sql(q).toPandas()
    pdf.attrs = {}   # serverless attaches Spark PlanMetrics here; pyarrow can't JSON-encode them
    pdf.to_parquet(f"{out}/{name}.parquet", index=False)
    print(f"{name}: {len(pdf):,} rows")
