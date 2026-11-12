-- Databricks notebook source
-- 04b · Plantable space per H3 cell (Databricks spatial SQL).
--   silver_obstacles_h3     every buffered building/road, assigned to the res-10 cells it touches
--   silver_plantable_h3     candidate cell minus the union of its obstacles: open ground polygon + m2
--   silver_plantable_spots  tree spots: res-13 cells whose center is on open ground, not in a paved-over
--                           or already-shaded block (NLCD at res 11), thinned to <= 1 per res-12 cell with >= 3 of 7 res-13 sub-cells open
-- Buffers and areas are computed in UTM 15N (EPSG:32615) so they are in meters.
USE CATALOG workspace;
USE SCHEMA heat;

-- COMMAND ----------

-- Same candidate rule as gold_planting_sites in 07
CREATE OR REPLACE TEMP VIEW cand AS
SELECT h3_10 FROM gold_hex_features
WHERE water_frac < 0.2 AND canopy_pct_2021 < 40 AND imperv_pct_2021 BETWEEN 10 AND 95;

-- COMMAND ----------

CREATE OR REPLACE TABLE silver_obstacles_h3
COMMENT 'Buffered Overture buildings and road/rail segments (no-plant zones, WKB in EPSG:32615) per candidate H3 res-10 cell they touch'
AS
WITH buf AS (
  SELECT kind, class, buffer_m,
         st_buffer(st_transform(st_geomfromwkb(geom_wkb, 4326), 32615), buffer_m) AS g
  FROM bronze_obstacles
  WHERE geom_wkb IS NOT NULL
),
cov AS (
  SELECT explode(h3_coverash3(st_asbinary(st_transform(g, 4326)), 10)) AS h3_10, kind, class,
         st_asbinary(g) AS g_wkb
  FROM buf
)
SELECT cov.h3_10, cov.kind, cov.class, cov.g_wkb
FROM cov JOIN cand c ON c.h3_10 = cov.h3_10;

-- COMMAND ----------

CREATE OR REPLACE TABLE silver_plantable_h3
COMMENT 'Open ground per candidate H3 res-10 cell: cell polygon minus buffered buildings and roads (Overture). open_wkb is EPSG:4326'
AS
WITH hexg AS (
  SELECT h3_10, st_transform(st_geomfromwkb(h3_boundaryaswkb(h3_10), 4326), 32615) AS hg FROM cand
),
obst AS (
  SELECT h3_10,
         st_union_agg(st_geomfromwkb(g_wkb, 32615)) AS og,
         count_if(kind = 'building') AS n_buildings,
         count_if(kind <> 'building') AS n_segments
  FROM silver_obstacles_h3 GROUP BY h3_10
),
open AS (
  SELECT h.h3_10, h.hg, coalesce(o.n_buildings, 0) AS n_buildings, coalesce(o.n_segments, 0) AS n_segments,
         CASE WHEN o.og IS NULL THEN h.hg ELSE st_difference(h.hg, o.og) END AS open_g
  FROM hexg h LEFT JOIN obst o USING (h3_10)
)
SELECT h3_10, n_buildings, n_segments,
       st_area(hg) AS cell_m2,
       st_area(open_g) AS open_m2,
       st_area(open_g) / st_area(hg) AS open_frac,
       st_asbinary(st_transform(open_g, 4326)) AS open_wkb
FROM open;

-- COMMAND ----------

CREATE OR REPLACE TABLE silver_plantable_spots
COMMENT 'Candidate tree spots: H3 res-13 cells on open ground (off buildings + setback, off roads), not in >90% impervious or >50% canopy blocks (NLCD at res 11), at most one per res-12 cell (~16 m apart) and only where >= 3 of its 7 res-13 sub-cells are open. spot_rank is a stable random order within each res-10 cell'
AS
WITH s AS (
  SELECT h3_10, explode(h3_polyfillash3(open_wkb, 13)) AS spot
  FROM silver_plantable_h3 WHERE open_m2 > 0
),
nlcd AS (
  SELECT h3_longlatash3(lon, lat, 11) AS h11,
         avg(CASE WHEN layer = 'imperv_2021' THEN value END) AS imperv,
         avg(CASE WHEN layer = 'canopy_2021' THEN value END) AS canopy
  FROM bronze_raster_pixels WHERE layer IN ('imperv_2021', 'canopy_2021')
  GROUP BY 1
),
ok AS (
  SELECT s.h3_10, s.spot
  FROM s LEFT JOIN nlcd n ON n.h11 = h3_toparent(s.spot, 11)
  WHERE h3_toparent(s.spot, 10) = s.h3_10          -- children of other cells poke across the edge
    AND coalesce(n.imperv, 0) <= 90 AND coalesce(n.canopy, 0) <= 50
),
thin AS (   -- a ~300 m2 patch holds a tree only if >= 3 of its 7 sub-cells (~130 m2) are open
  SELECT h3_10, min_by(spot, xxhash64(spot)) AS spot FROM ok GROUP BY h3_10, h3_toparent(spot, 12)
  HAVING count(*) >= 3
)
SELECT h3_10, spot,
       row_number() OVER (PARTITION BY h3_10 ORDER BY xxhash64(spot)) AS spot_rank,
       st_x(st_geomfromwkb(h3_centeraswkb(spot))) AS lon,
       st_y(st_geomfromwkb(h3_centeraswkb(spot))) AS lat
FROM thin;

-- COMMAND ----------

-- How much of a typical candidate cell is open ground, and how many tree spots it holds
SELECT CAST(floor(g.imperv_pct_2021 / 20) * 20 AS INT) AS imperv_band, count(*) AS cells,
       round(avg(p.open_frac), 2) AS open_frac, round(avg(p.n_buildings), 1) AS buildings,
       round(avg(coalesce(sp.n, 0)), 1) AS spots
FROM silver_plantable_h3 p
JOIN gold_hex_features g USING (h3_10)
LEFT JOIN (SELECT h3_10, count(*) n FROM silver_plantable_spots GROUP BY 1) sp USING (h3_10)
GROUP BY 1 ORDER BY 1;
