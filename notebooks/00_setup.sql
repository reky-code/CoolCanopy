-- Databricks notebook source
-- Idempotent setup: schema + volumes for the heat-trees project (Free Edition: `workspace` catalog).
CREATE SCHEMA IF NOT EXISTS workspace.heat
  COMMENT 'Urban heat islands x street trees (TAMU Datathon)';

-- COMMAND ----------

CREATE VOLUME IF NOT EXISTS workspace.heat.raw COMMENT 'Raw downloads (CSV, GeoTIFF, GeoParquet)';
CREATE VOLUME IF NOT EXISTS workspace.heat.artifacts COMMENT 'Model artifacts, effect tables, exports';
