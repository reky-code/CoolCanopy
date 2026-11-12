"""Local smoke test: STAC search + read one decimated scene over the city bbox."""
import sys
import time

sys.path.insert(0, "src")
from heat import landsat  # noqa: E402
from heat.config import load_config  # noqa: E402

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "conf/city_houston.yaml")
cfg["landsat"]["years"] = [2023]
t = time.time()
rows = landsat.search_scenes(cfg)
scenes = landsat.pick_scenes(rows, 3)
print(f"search: {len(rows)} scenes in 2023 summer ({time.time()-t:.1f}s)")
for s in scenes:
    print(" ", s["scene_id"], s["scene_date"], s["platform"], s["cloud_cover"])

t = time.time()
lc = cfg["landsat"]
df = landsat.read_scene(scenes[0], cfg["bbox"], lc["st_scale"], lc["st_offset"], decimate=4)
print(f"read {scenes[0]['scene_id']}: {len(df):,} px (1/16 sample) in {time.time()-t:.1f}s")
land = df[~df.is_water]
print(f"  land LST C: p05={land.lst_c.quantile(.05):.1f} median={land.lst_c.median():.1f} "
      f"p95={land.lst_c.quantile(.95):.1f}; water frac={df.is_water.mean():.2f}")
print(f"  lon [{df.lon.min():.3f},{df.lon.max():.3f}] lat [{df.lat.min():.3f},{df.lat.max():.3f}]")
