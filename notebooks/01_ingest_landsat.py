# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest Landsat surface temperature (bronze)
# MAGIC STAC search on Planetary Computer → one Spark task per scene reads the city window
# MAGIC (`lwir11` = ST_B10, `qa_pixel`) → cloud/shadow mask → °C → `bronze_landsat_pixels`.

# COMMAND ----------

# MAGIC %pip install -q pystac-client planetary-computer rasterio pyproj pyyaml

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
from heat import landsat

cfg = load_config(os.path.join(bundle_root, dbutils.widgets.get("city_config")))
lc = cfg["landsat"]
T = cfg["fq_schema"]
print(cfg["display_name"], cfg["bbox"], "->", T)

# COMMAND ----------

rows = landsat.search_scenes(cfg)
scenes = landsat.pick_scenes(rows, lc["max_scenes"], lc.get("skip_tiles"))
print(f"{len(rows)} candidate scenes, using {len(scenes)}")

scenes_df = spark.createDataFrame(scenes)
scenes_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{T}.bronze_landsat_scenes")
display(scenes_df.select("scene_id", "scene_date", "platform", "cloud_cover", "wrs_path", "wrs_row"))

# COMMAND ----------

import pyspark.sql.functions as F

bbox, scale, offset = cfg["bbox"], lc["st_scale"], lc["st_offset"]
schema = ("scene_id string, scene_date string, platform string, lon double, lat double, "
          "lst_c float, is_water boolean")

# Workers don't see the driver's sys.path, and pickling the module by value drags in
# Databricks' notebook import hooks (ModuleNotFoundError: dbruntime). So ship the module
# *source* as a string and exec it in a fresh namespace on each worker.
with open(landsat.__file__, encoding="utf-8") as f:
    landsat_src = f.read()

def read_partition(batches):
    import types
    mod = types.ModuleType("heat_landsat_worker")
    exec(compile(landsat_src, "heat/landsat.py", "exec"), mod.__dict__)
    yield from mod.read_scenes_batches(batches, bbox, scale, offset)

pixels = (scenes_df.repartition(len(scenes))
          .mapInPandas(read_partition, schema)
          .withColumn("scene_date", F.to_date("scene_date")))

(pixels.write.mode("overwrite").option("overwriteSchema", "true")
       .partitionBy("scene_id").saveAsTable(f"{T}.bronze_landsat_pixels"))

# COMMAND ----------

display(spark.sql(f"""
  SELECT scene_id, scene_date, count(*) AS n_px, round(avg(lst_c),1) AS mean_c,
         round(percentile_approx(lst_c, 0.5),1) AS median_c, round(avg(cast(is_water AS int)),3) AS water_frac
  FROM {T}.bronze_landsat_pixels GROUP BY ALL ORDER BY scene_date"""))
