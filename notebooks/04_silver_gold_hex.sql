-- Databricks notebook source
-- 04 · Silver context on H3 + first gold feature table, clipped to city limits.
--   silver_city_h3       res-10 cells whose center lies in full-purpose city limits, with council district
--   silver_landcover_h3  NLCD canopy % (2011/2016/2021) and impervious % per cell
--   gold_hex_features    heat anomaly + land cover + neighborhood rings (k=1, k=2), city cells only
USE CATALOG workspace;
USE SCHEMA heat;

-- COMMAND ----------

CREATE OR REPLACE TABLE silver_city_h3
COMMENT 'H3 res-10 cells inside full-purpose city limits (center containment), with council district (2022 boundaries) and Super Neighborhood'
AS
WITH city AS (
  SELECT DISTINCT explode(h3_polyfillash3(polygon_geojson, 10)) AS h3_10
  FROM bronze_boundaries WHERE kind = 'city_limits'
),
dist AS (
  SELECT name AS district, explode(h3_polyfillash3(polygon_geojson, 10)) AS h3_10
  FROM bronze_boundaries WHERE kind = 'council_district'
),
snbhd AS (   -- names arrive upper-case ("GULFTON"); store them title-cased for display
  SELECT initcap(lower(trim(name))) AS super_nbhd, explode(h3_polyfillash3(polygon_geojson, 10)) AS h3_10
  FROM bronze_boundaries WHERE kind = 'super_neighborhood'
)
SELECT c.h3_10, min(d.district) AS district, min(s.super_nbhd) AS super_nbhd
FROM city c
LEFT JOIN dist d USING (h3_10)
LEFT JOIN snbhd s USING (h3_10)
GROUP BY c.h3_10;

-- COMMAND ----------

CREATE OR REPLACE TABLE silver_landcover_h3
COMMENT 'NLCD 30 m percent layers averaged per H3 res-10 cell'
AS
SELECT h3_longlatash3(lon, lat, 10) AS h3_10,
       avg(CASE WHEN layer = 'canopy_2021' THEN value END) AS canopy_pct_2021,
       avg(CASE WHEN layer = 'canopy_2016' THEN value END) AS canopy_pct_2016,
       avg(CASE WHEN layer = 'canopy_2011' THEN value END) AS canopy_pct_2011,
       avg(CASE WHEN layer = 'imperv_2021' THEN value END) AS imperv_pct_2021,
       count_if(layer = 'canopy_2021')                      AS n_px
FROM bronze_raster_pixels
GROUP BY 1;

-- COMMAND ----------

CREATE OR REPLACE TABLE gold_hex_features
COMMENT 'Per-cell features for the heat model: Houston city limits, H3 res 10'
AS
WITH base AS (
  SELECT c.h3_10, c.district, c.super_nbhd, l.canopy_pct_2021, l.canopy_pct_2016, l.canopy_pct_2011,
         l.imperv_pct_2021
  FROM silver_city_h3 c JOIN silver_landcover_h3 l USING (h3_10)
),
nb AS (   -- neighborhood rings: every cell paired with its k<=2 neighbors
  SELECT b.h3_10, n.nb, h3_distance(b.h3_10, n.nb) AS k
  FROM base b LATERAL VIEW explode(h3_kring(b.h3_10, 2)) n AS nb
),
rings AS (
  SELECT nb.h3_10,
         avg(CASE WHEN k <= 1 THEN l.canopy_pct_2021 END) AS canopy_k1,
         avg(CASE WHEN k <= 2 THEN l.canopy_pct_2021 END) AS canopy_k2,
         avg(CASE WHEN k <= 1 THEN l.imperv_pct_2021 END) AS imperv_k1,
         avg(CASE WHEN k <= 2 THEN l.imperv_pct_2021 END) AS imperv_k2
  FROM nb JOIN silver_landcover_h3 l ON l.h3_10 = nb.nb
  GROUP BY nb.h3_10
)
SELECT b.*, r.canopy_k1, r.canopy_k2, r.imperv_k1, r.imperv_k2,
       s.lst_anom_c, s.lst_anom_sd_c, s.lst_mean_c, s.water_frac, s.n_scenes,
       h3_h3tostring(b.h3_10) AS h3_10_str, h3_centeraswkt(b.h3_10) AS center_wkt
FROM base b
JOIN rings r USING (h3_10)
JOIN silver_lst_h3 s USING (h3_10)
WHERE s.water_frac < 0.5;

-- COMMAND ----------

-- First look: heat anomaly by canopy band, holding impervious roughly fixed (40-70%).
SELECT CAST(floor(canopy_pct_2021 / 10) * 10 AS INT) AS canopy_band_pct,
       count(*) AS n_hex, round(avg(lst_anom_c), 2) AS mean_anom_c,
       round(avg(imperv_pct_2021), 1) AS mean_imperv_pct
FROM gold_hex_features
WHERE imperv_pct_2021 BETWEEN 40 AND 70
GROUP BY 1 ORDER BY 1;
