# Databricks notebook source
# MAGIC %md
# MAGIC # 03b · Ingest obstacles for tree placement (bronze)
# MAGIC Building footprints and road/rail segments from **Overture Maps** (public GeoParquet on S3).
# MAGIC DuckDB on the serverless driver reads only the row groups that overlap the city bbox and
# MAGIC writes Parquet to the `raw` volume; Spark loads it as `bronze_obstacles`
# MAGIC (`kind, class, buffer_m, geom_wkb`). Buffers turn into no-plant zones in `04b`.

# COMMAND ----------

# MAGIC %pip install -q duckdb pyyaml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
import time

nb_dir = os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext()
                         .notebookPath().get())
bundle_root = os.path.abspath(os.path.join("/Workspace" + nb_dir, ".."))
sys.path.insert(0, os.path.join(bundle_root, "src"))
dbutils.widgets.text("city_config", "conf/city_houston.yaml")

from heat.config import load_config
from heat import footprints

cfg = load_config(os.path.join(bundle_root, dbutils.widgets.get("city_config")))
T, city, bbox, fp = cfg["fq_schema"], cfg["city"], cfg["bbox"], cfg["footprints"]
out = f"{cfg['raw_volume']}/{city}/overture/{fp['release']}"
os.makedirs(out, exist_ok=True)
print("Overture", fp["release"], "->", out)

# COMMAND ----------

import shutil

import duckdb

con = duckdb.connect()
con.sql("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
con.sql(f"SET s3_region='{fp.get('s3_region', 'us-west-2')}'; SET threads=8;")

paths = []
for name, q in footprints.overture_queries(fp, bbox).items():
    t = time.time()
    local = f"/tmp/{name}.parquet"   # DuckDB writes locally, then copy into the volume
    con.execute(f"COPY ({q}) TO '{local}' (FORMAT parquet, COMPRESSION zstd)")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{local}')").fetchall()[0][0]
    dst = f"{out}/{name}.parquet"
    shutil.copyfile(local, dst)   # serverless blocks dbutils.fs.cp from local disk; FUSE copy works
    paths.append(dst)
    print(f"{name}: {n:,} features in {time.time() - t:.0f}s")

# COMMAND ----------

(spark.read.parquet(*paths)
      .selectExpr("kind", "class", "CAST(buffer_m AS DOUBLE) AS buffer_m", "geom_wkb")
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(f"{T}.bronze_obstacles"))
spark.sql(f"COMMENT ON TABLE {T}.bronze_obstacles IS 'Overture Maps {fp['release']} buildings + road/rail "
          f"segments in the city bbox (WKB, EPSG:4326), with a no-plant buffer in meters per feature'")
display(spark.sql(f"""SELECT kind, class, count(*) n, avg(buffer_m) buffer_m FROM {T}.bronze_obstacles
                      GROUP BY ALL ORDER BY n DESC"""))
