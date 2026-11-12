# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Heat model: OLS headline + monotone LightGBM with spatial CV (MLflow, UC registry)
# MAGIC - **OLS** with SEs clustered by H3 res-6 parent → the interpretable "°C per +10% canopy"
# MAGIC - **LightGBM**: canopy can only cool and pavement can only heat (monotone constraints);
# MAGIC   5-fold **spatial** CV (hold out whole res-6 parents) vs random CV to show leakage
# MAGIC - SHAP-style contributions via LightGBM `pred_contrib` (no extra dependency)
# MAGIC - Registers `workspace.heat.heat_model`; also saves boosters to the artifacts volume for the simulator

# COMMAND ----------

# MAGIC %pip install -q lightgbm scikit-learn statsmodels pyyaml "mlflow>=2.16"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import sys

import mlflow
import numpy as np
import pandas as pd

nb_dir = os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext()
                         .notebookPath().get())
bundle_root = os.path.abspath(os.path.join("/Workspace" + nb_dir, ".."))
sys.path.insert(0, os.path.join(bundle_root, "src"))
dbutils.widgets.text("city_config", "conf/city_houston.yaml")

from heat.config import load_config
from heat import model as hm

cfg = load_config(os.path.join(bundle_root, dbutils.widgets.get("city_config")))
T, city = cfg["fq_schema"], cfg["city"]
art = f"/Volumes/{cfg['catalog']}/{cfg['schema']}/artifacts/{city}/models"
os.makedirs(art, exist_ok=True)

df = spark.sql(f"""
  SELECT *, h3_toparent(h3_10, 6) AS parent_r6,
         CAST(regexp_extract(center_wkt, 'POINT\\\\(([-0-9.]+) ', 1) AS DOUBLE) AS lon,
         CAST(regexp_extract(center_wkt, ' ([-0-9.]+)\\\\)', 1) AS DOUBLE) AS lat
  FROM {T}.gold_hex_features
  WHERE lst_anom_c IS NOT NULL AND n_scenes >= 3
""").toPandas()
df = df.dropna(subset=hm.FEATURES + [hm.TARGET])
print(f"{len(df):,} cells, {df.parent_r6.nunique()} res-6 groups")

# COMMAND ----------

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{user}/heat-trees-{city}")

with mlflow.start_run(run_name=f"heat_model_{city}") as run:
    ols = hm.ols_baseline(df)
    print(ols.round(4))
    mlflow.log_metric("ols_r2", ols.attrs["r2"])
    for term in ("canopy_pct_2021", "canopy_k2", "imperv_pct_2021"):
        mlflow.log_metric(f"ols_{term}_per10_c", float(ols.loc[term, "per_10pct_c"]))

    res = hm.train_spatial_cv(df)
    print(res.metrics)
    print(res.fold_metrics)
    mlflow.log_params({"features": ",".join(hm.FEATURES), "cv": "GroupKFold(5) by H3 res-6",
                       "num_rounds": 600, **{k: v for k, v in hm.DEFAULT_PARAMS.items()
                                             if k in ("learning_rate", "num_leaves", "min_data_in_leaf")}})
    mlflow.log_metrics(res.metrics)

    # SHAP-style contributions on a sample
    samp = df.sample(min(20000, len(df)), random_state=7)
    contrib = res.full_model.predict(samp[hm.FEATURES].to_numpy(float), pred_contrib=True)
    imp = pd.DataFrame({"feature": hm.FEATURES,
                        "mean_abs_contrib_c": np.abs(contrib[:, :-1]).mean(axis=0)}
                       ).sort_values("mean_abs_contrib_c", ascending=False)
    print(imp)
    canopy_cols = [hm.FEATURES.index(c) for c in ("canopy_pct_2021", "canopy_k1", "canopy_k2")]
    samp = samp.assign(canopy_contrib_c=contrib[:, canopy_cols].sum(axis=1),
                       canopy_band=(samp.canopy_pct_2021 // 10 * 10).astype(int))
    dep = samp.groupby("canopy_band").agg(n=("h3_10", "size"),
                                          canopy_contrib_c=("canopy_contrib_c", "mean")).reset_index()
    print(dep)

    # Register the point model in Unity Catalog
    from mlflow.models import infer_signature
    X_ex = df[hm.FEATURES].head(50)
    mlflow.lightgbm.log_model(res.full_model, "model",
                              signature=infer_signature(X_ex, res.full_model.predict(X_ex.to_numpy(float))),
                              input_example=X_ex.head(5),
                              registered_model_name=f"{T}.heat_model")
    run_id = run.info.run_id

# Boosters for the simulator (point model + spatial-fold ensemble for uncertainty).
# LightGBM seeks while writing, which UC Volumes (FUSE) don't allow: write locally, then copy.
import shutil
import tempfile

tmp = tempfile.mkdtemp()
boosters = {"full.txt": res.full_model, **{f"fold_{k}.txt": m for k, m in enumerate(res.fold_models)}}
for name, m in boosters.items():
    m.save_model(f"{tmp}/{name}")
    shutil.copyfile(f"{tmp}/{name}", f"{art}/{name}")
with open(f"{art}/features.json", "w") as f:
    json.dump({"features": hm.FEATURES, "mlflow_run_id": run_id}, f)

# COMMAND ----------

metrics_rows = [{"metric": k, "value": float(v)} for k, v in res.metrics.items()]
metrics_rows += [{"metric": "ols_r2", "value": ols.attrs["r2"]}]
metrics_rows += [{"metric": f"ols_{t}_per10_c", "value": float(ols.loc[t, "per_10pct_c"])}
                 for t in ols.index if t != "const"]
metrics_rows += [{"metric": f"ols_{t}_per10_se_c", "value": float(ols.loc[t, "se"] * 10)}
                 for t in ols.index if t != "const"]
spark.createDataFrame(pd.DataFrame(metrics_rows)).write.mode("overwrite") \
     .option("overwriteSchema", "true").saveAsTable(f"{T}.gold_model_metrics")
spark.createDataFrame(imp).write.mode("overwrite").option("overwriteSchema", "true") \
     .saveAsTable(f"{T}.gold_model_importance")
spark.createDataFrame(dep).write.mode("overwrite").option("overwriteSchema", "true") \
     .saveAsTable(f"{T}.gold_model_canopy_dependence")

# Out-of-fold predictions per cell (residual maps help spot data problems)
spark.createDataFrame(pd.DataFrame({"h3_10": df.h3_10.values, "pred_oof_c": res.oof_pred,
                                    "resid_c": df[hm.TARGET].values - res.oof_pred})) \
     .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{T}.gold_model_oof")
display(spark.table(f"{T}.gold_model_metrics"))
