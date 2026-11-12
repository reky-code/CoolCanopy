# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Ingest context layers (bronze)
# MAGIC - City limits, council districts (2022 boundaries) and Super Neighborhoods from the City of
# MAGIC   Houston ArcGIS services → `bronze_boundaries`
# MAGIC - NLCD tree canopy (several years) and impervious surface (30 m) from MRLC WCS → GeoTIFF in the
# MAGIC   `raw` volume → `bronze_raster_pixels` (one row per pixel; H3 indexing happens in SQL)

# COMMAND ----------

# MAGIC %pip install -q rasterio pyproj pyyaml requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

nb_dir = os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext()
                         .notebookPath().get())
bundle_root = os.path.abspath(os.path.join("/Workspace" + nb_dir, ".."))
sys.path.insert(0, os.path.join(bundle_root, "src"))

dbutils.widgets.text("city_config", "conf/city_houston.yaml")

from heat.config import load_config
from heat import context

cfg = load_config(os.path.join(bundle_root, dbutils.widgets.get("city_config")))
T, city, bbox = cfg["fq_schema"], cfg["city"], cfg["bbox"]
raw = f"{cfg['raw_volume']}/{city}"
print(cfg["display_name"], "->", T, "| raw:", raw)

# COMMAND ----------

import pandas as pd

rows = []
for kind, b in cfg["boundaries"].items():
    if not isinstance(b, dict) or "url" not in b:     # e.g. district_ids
        continue
    for p in context.fetch_arcgis_polygons(b["url"], b["where"], b["name_field"]):
        rows.append({"city": city, "kind": kind, **p})
bdf = pd.DataFrame(rows)
print(bdf.groupby("kind").size())
(spark.createDataFrame(bdf).write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(f"{T}.bronze_boundaries"))

# COMMAND ----------

wcs = cfg["rasters"]["wcs_url"]
parts = []
for layer, coverage in cfg["rasters"]["layers"].items():
    tif = context.fetch_wcs_geotiff(wcs, coverage, bbox, f"{raw}/nlcd/{layer}.tif")
    pts = context.raster_to_points(tif, bbox)
    pts.insert(0, "layer", layer)
    pq = f"{raw}/nlcd/{layer}.parquet"
    pts.to_parquet(pq, index=False)
    parts.append(pq)
    print(f"{layer}: {len(pts):,} px, mean {pts.value.mean():.1f}")

(spark.read.parquet(*parts)
      .selectExpr("layer", "lon", "lat", "CAST(value AS SMALLINT) AS value")
      .write.mode("overwrite").option("overwriteSchema", "true")
      .partitionBy("layer").saveAsTable(f"{T}.bronze_raster_pixels"))

# COMMAND ----------

display(spark.sql(f"SELECT layer, count(*) n_px, round(avg(value),1) mean_pct FROM {T}.bronze_raster_pixels GROUP BY layer ORDER BY layer"))
