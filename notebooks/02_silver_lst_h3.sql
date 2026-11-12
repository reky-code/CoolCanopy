-- Databricks notebook source
-- 02 · Silver: Landsat pixels -> H3 res-10 heat anomaly.
-- Per scene: anomaly = hex mean LST - that scene's median over REFERENCE hexes.
-- Reference hexes are covered by every WRS path in the scene set. A city that straddles
-- paths (Houston: 025 and 026) has scenes that see different parts of town. Taking each
-- scene's median over its own footprint would mix "hotter day" with "hotter half of the
-- city", so we normalize on the shared overlap instead. For single-path cities every
-- hex is a reference hex.
-- Then average anomalies across scenes, keeping the spread as an uncertainty measure.
USE CATALOG workspace;
USE SCHEMA heat;

-- COMMAND ----------

CREATE OR REPLACE TABLE silver_lst_scene_h3 AS
WITH px AS (
  SELECT p.scene_id, p.scene_date, s.wrs_path, h3_longlatash3(p.lon, p.lat, 10) AS h3_10,
         p.lst_c, p.is_water
  FROM bronze_landsat_pixels p JOIN bronze_landsat_scenes s USING (scene_id)
),
n_paths AS (SELECT count(DISTINCT wrs_path) AS n FROM bronze_landsat_scenes),
ref_hex AS (
  SELECT h3_10 FROM px WHERE NOT is_water
  GROUP BY h3_10
  HAVING count(DISTINCT wrs_path) = (SELECT n FROM n_paths)
),
med AS (
  SELECT px.scene_id, percentile_approx(px.lst_c, 0.5) AS ref_median_c, count(*) AS n_ref_px
  FROM px JOIN ref_hex USING (h3_10)
  WHERE NOT px.is_water GROUP BY px.scene_id
)
SELECT px.scene_id,
       any_value(px.scene_date)                              AS scene_date,
       any_value(px.wrs_path)                                AS wrs_path,
       px.h3_10,
       avg(CASE WHEN NOT is_water THEN lst_c END)            AS lst_c,
       avg(CASE WHEN NOT is_water THEN lst_c END) - any_value(med.ref_median_c) AS lst_anom_c,
       avg(CAST(is_water AS INT))                            AS water_frac,
       count(*)                                              AS n_px
FROM px JOIN med USING (scene_id)
WHERE med.n_ref_px >= 1000        -- drop scenes that barely touch the shared overlap
GROUP BY px.scene_id, px.h3_10;

-- COMMAND ----------

CREATE OR REPLACE TABLE silver_lst_h3
COMMENT 'Summer surface-temperature anomaly per H3 res-10 cell (deg C vs scene reference median), Landsat 8/9 ~10:30am'
AS
SELECT h3_10,
       h3_h3tostring(h3_10)          AS h3_10_str,
       h3_centeraswkt(h3_10)         AS center_wkt,
       avg(lst_anom_c)               AS lst_anom_c,
       stddev(lst_anom_c)            AS lst_anom_sd_c,
       avg(lst_c)                    AS lst_mean_c,
       avg(water_frac)               AS water_frac,
       count(*)                      AS n_scenes,
       count(DISTINCT wrs_path)      AS n_paths,
       sum(n_px)                     AS n_px
FROM silver_lst_scene_h3
WHERE lst_c IS NOT NULL AND n_px >= 5
GROUP BY h3_10
HAVING count(*) >= 3;

-- COMMAND ----------

-- Sanity check: mean anomaly by path. Paths cover different areas, so this need not be 0,
-- but a path offset of several degrees would mean the normalization failed.
SELECT wrs_path, count(*) AS n_hex_scene, round(avg(lst_anom_c), 2) AS mean_anom_c
FROM silver_lst_scene_h3 GROUP BY wrs_path ORDER BY wrs_path;

-- COMMAND ----------

SELECT count(*) AS n_hex, round(avg(n_scenes),1) AS avg_scenes,
       round(percentile_approx(lst_anom_c, 0.05),2) AS p05_anom_c,
       round(percentile_approx(lst_anom_c, 0.95),2) AS p95_anom_c,
       round(avg(lst_anom_sd_c),2) AS mean_sd_c
FROM silver_lst_h3;
