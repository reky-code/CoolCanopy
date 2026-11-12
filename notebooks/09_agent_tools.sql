-- Databricks notebook source
-- 09 · Unity Catalog functions used as agent tools (read-only, governed, discoverable).
-- The optimizer itself runs inside the agent (Python); these answer lookups from the gold tables.
USE CATALOG workspace;
USE SCHEMA heat;

-- COMMAND ----------

CREATE OR REPLACE FUNCTION site_details(h3_cell STRING COMMENT 'H3 res-10 cell id as hex string, e.g. 8a446ca8549ffff')
RETURNS TABLE (h3_cell STRING, district STRING, super_neighborhood STRING, lst_anom_c DOUBLE, canopy_pct DOUBLE,
               imperv_pct DOUBLE, residents DOUBLE, age65 DOUBLE, svi_rank DOUBLE, is_candidate BOOLEAN,
               capacity INT, lon DOUBLE, lat DOUBLE)
COMMENT 'Facts about one H3 cell in Houston: council district, Super Neighborhood, surface heat anomaly (deg C vs city median, Landsat summer ~10:30am), NLCD canopy and impervious %, residents, residents 65+, CDC SVI percentile, and whether it is a planting candidate.'
RETURN
  SELECT h3_h3tostring(g.h3_10), g.district, g.super_nbhd, g.lst_anom_c, g.canopy_pct_2021, g.imperv_pct_2021,
         p.residents, p.age65, p.svi_rank, s.site_hex IS NOT NULL, s.capacity,
         CAST(regexp_extract(g.center_wkt, 'POINT\\(([-0-9.]+) ', 1) AS DOUBLE),
         CAST(regexp_extract(g.center_wkt, ' ([-0-9.]+)\\)', 1) AS DOUBLE)
  FROM gold_hex_features g
  LEFT JOIN silver_pop_h3 p USING (h3_10)
  LEFT JOIN gold_planting_sites s ON s.site_hex = g.h3_10
  WHERE g.h3_10 = h3_stringtoh3(lower(trim(site_details.h3_cell)));

-- COMMAND ----------

CREATE OR REPLACE FUNCTION nearest_cell(lat DOUBLE COMMENT 'latitude, WGS84', lon DOUBLE COMMENT 'longitude, WGS84')
RETURNS STRING
COMMENT 'Convert a latitude/longitude in Houston to the H3 res-10 cell id the other tools use.'
RETURN h3_h3tostring(h3_longlatash3(lon, lat, 10));

-- COMMAND ----------

CREATE OR REPLACE FUNCTION district_summary(district_id STRING COMMENT 'Houston council district letter A-K, or ALL')
RETURNS TABLE (district STRING, n_cells BIGINT, mean_anom_c DOUBLE, mean_canopy_pct DOUBLE,
               mean_imperv_pct DOUBLE, residents DOUBLE, age65 DOUBLE, vulnerable DOUBLE,
               hot_cells BIGINT, candidate_sites BIGINT)
COMMENT 'Heat, canopy, pavement and population summary per Houston council district. hot_cells = cells hotter than +3 deg C vs the city median.'
RETURN
  SELECT g.district, count(*), avg(g.lst_anom_c), avg(g.canopy_pct_2021), avg(g.imperv_pct_2021),
         sum(p.residents), sum(p.age65), sum(p.vulnerable),
         count_if(g.lst_anom_c > 3), count(s.site_hex)
  FROM gold_hex_features g
  LEFT JOIN silver_pop_h3 p USING (h3_10)
  LEFT JOIN gold_planting_sites s ON s.site_hex = g.h3_10
  WHERE upper(district_summary.district_id) = 'ALL' OR g.district = upper(district_summary.district_id)
  GROUP BY g.district;

-- COMMAND ----------

CREATE OR REPLACE FUNCTION neighborhood_summary(name STRING COMMENT 'Super Neighborhood name or part of it, e.g. Gulfton or Uptown; ALL for every one')
RETURNS TABLE (super_neighborhood STRING, district STRING, n_cells BIGINT, mean_anom_c DOUBLE,
               mean_canopy_pct DOUBLE, mean_imperv_pct DOUBLE, residents DOUBLE, age65 DOUBLE,
               vulnerable DOUBLE, hot_cells BIGINT, candidate_sites BIGINT)
COMMENT 'Heat, canopy, pavement and population summary per Houston Super Neighborhood (the city''s 88 neighborhood planning areas). district = council district with most of its residents. hot_cells = cells hotter than +3 deg C vs the city median.'
RETURN
  SELECT g.super_nbhd, max_by(g.district, coalesce(p.residents, 0)), count(*), avg(g.lst_anom_c),
         avg(g.canopy_pct_2021), avg(g.imperv_pct_2021),
         sum(p.residents), sum(p.age65), sum(p.vulnerable),
         count_if(g.lst_anom_c > 3), count(s.site_hex)
  FROM gold_hex_features g
  LEFT JOIN silver_pop_h3 p USING (h3_10)
  LEFT JOIN gold_planting_sites s ON s.site_hex = g.h3_10
  WHERE g.super_nbhd IS NOT NULL
    AND (upper(neighborhood_summary.name) = 'ALL'
         OR upper(g.super_nbhd) LIKE concat('%', upper(trim(neighborhood_summary.name)), '%'))
  GROUP BY g.super_nbhd;

-- COMMAND ----------

CREATE OR REPLACE FUNCTION hottest_cells(district_id STRING COMMENT 'council district letter A-K, a Super Neighborhood name (e.g. Gulfton), or ALL',
                                         min_age65 DOUBLE COMMENT 'minimum residents aged 65+ in the cell (0 for any)',
                                         max_canopy_pct DOUBLE COMMENT 'only cells with canopy at or below this %',
                                         n INT COMMENT 'how many cells to return (max 50)')
RETURNS TABLE (h3_cell STRING, district STRING, super_neighborhood STRING, lst_anom_c DOUBLE, canopy_pct DOUBLE,
               residents DOUBLE, age65 DOUBLE, svi_rank DOUBLE)
COMMENT 'Hottest populated cells, optionally filtered by council district or Super Neighborhood, elderly residents and low canopy. Use for questions like which hot places with many elderly residents have the least shade.'
RETURN
  SELECT h3_cell, district, super_neighborhood, lst_anom_c, canopy_pct, residents, age65, svi_rank
  FROM (
    SELECT h3_h3tostring(g.h3_10) AS h3_cell, g.district, g.super_nbhd AS super_neighborhood, g.lst_anom_c,
           g.canopy_pct_2021 AS canopy_pct, p.residents, p.age65, p.svi_rank,
           row_number() OVER (ORDER BY g.lst_anom_c DESC) AS rk
    FROM gold_hex_features g JOIN silver_pop_h3 p USING (h3_10)
    WHERE (upper(hottest_cells.district_id) = 'ALL' OR g.district = upper(hottest_cells.district_id)
           OR (length(trim(hottest_cells.district_id)) > 2
               AND upper(g.super_nbhd) LIKE concat('%', upper(trim(hottest_cells.district_id)), '%')))
      AND p.age65 >= hottest_cells.min_age65 AND g.canopy_pct_2021 <= hottest_cells.max_canopy_pct
      AND p.residents > 0
  ) ranked
  WHERE rk <= least(hottest_cells.n, 50)
  ORDER BY rk;

-- COMMAND ----------

CREATE OR REPLACE FUNCTION compare_strategies(n INT COMMENT 'plan size in trees: 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000 or 5000')
RETURNS TABLE (strategy STRING, n_trees INT, residents_benefiting BIGINT, vulnerable_benefiting BIGINT,
               degree_vulnerable_c DOUBLE, hot_decile_hexes_reached BIGINT,
               mean_cooling_c DOUBLE, mean_cooling_lo_c DOUBLE, mean_cooling_hi_c DOUBLE)
COMMENT 'Precomputed comparison of planting strategies (random, even per district, hottest_first, cooling, vulnerability) at a given number of trees.'
RETURN
  SELECT strategy, n_trees, residents_benefiting, vulnerable_benefiting, degree_vulnerable_c,
         hot_decile_hexes_reached, mean_cooling_benefiting_hexes_c, mean_cooling_lo_c, mean_cooling_hi_c
  FROM gold_plans WHERE n_trees = compare_strategies.n
  ORDER BY degree_vulnerable_c DESC;

-- COMMAND ----------

CREATE OR REPLACE FUNCTION model_card()
RETURNS TABLE (metric STRING, value DOUBLE)
COMMENT 'Heat model quality and headline effects: spatial-CV R2/MAE, random-CV R2, OLS deg C per +10 pct canopy/impervious with clustered SEs.'
RETURN SELECT metric, value FROM gold_model_metrics;

-- COMMAND ----------

-- One wide, well-commented view for Genie (natural-language questions from planners).
CREATE OR REPLACE VIEW genie_cells
COMMENT 'One row per H3 res-10 cell (~15,000 m2) inside Houston city limits: summer surface heat, tree canopy, pavement, residents and vulnerability.'
AS
SELECT h3_h3tostring(g.h3_10)                    AS cell_id,
       g.district                                AS council_district,
       g.super_nbhd                              AS super_neighborhood,
       round(g.lst_anom_c, 2)                    AS surface_heat_vs_city_c,
       round(g.lst_anom_c * 1.8, 2)              AS surface_heat_vs_city_f,
       round(g.lst_mean_c, 1)                    AS surface_temp_mean_c,
       round(g.canopy_pct_2021, 1)               AS tree_canopy_pct,
       round(g.canopy_pct_2021 - g.canopy_pct_2011, 1) AS canopy_change_2011_2021_pct,
       round(g.imperv_pct_2021, 1)               AS impervious_pct,
       round(coalesce(p.residents, 0))           AS residents,
       round(coalesce(p.age65, 0))               AS residents_65_plus,
       round(coalesce(p.age17, 0))               AS residents_17_under,
       round(coalesce(p.pov150, 0))              AS residents_below_150pct_poverty,
       round(p.svi_rank, 3)                      AS social_vulnerability_percentile,
       s.site_hex IS NOT NULL                    AS is_planting_candidate,
       CAST(regexp_extract(g.center_wkt, ' ([-0-9.]+)\\)', 1) AS DOUBLE)       AS lat,
       CAST(regexp_extract(g.center_wkt, 'POINT\\(([-0-9.]+) ', 1) AS DOUBLE) AS lon
FROM gold_hex_features g
LEFT JOIN silver_pop_h3 p USING (h3_10)
LEFT JOIN gold_planting_sites s ON s.site_hex = g.h3_10;

-- COMMAND ----------

SELECT * FROM district_summary('ALL') ORDER BY mean_anom_c DESC;
