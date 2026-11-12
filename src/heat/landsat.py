"""Landsat Collection 2 Level-2 surface temperature: scene search + windowed pixel reads.

Pure functions so they can be tested locally and shipped to Spark workers by value.
Hrefs are stored unsigned and signed on the worker (Planetary Computer SAS tokens expire).
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

# QA_PIXEL bits (Landsat C2): 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow,
# 5 snow, 6 clear, 7 water
_BAD_BITS = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
_WATER_BIT = 1 << 7

PIXEL_COLUMNS = ["scene_id", "scene_date", "platform", "lon", "lat", "lst_c", "is_water"]


def qa_masks(qa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (usable, is_water) boolean arrays from a QA_PIXEL array."""
    qa = qa.astype(np.uint16)
    usable = (qa & _BAD_BITS) == 0
    water = (qa & _WATER_BIT) != 0
    return usable, water


def dn_to_celsius(dn: np.ndarray, scale: float, offset: float) -> np.ndarray:
    return dn.astype(np.float64) * scale + offset - 273.15


def search_scenes(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """STAC search for clear summer scenes over the city bbox; returns unsigned hrefs."""
    from pystac_client import Client

    lc = cfg["landsat"]
    client = Client.open(lc["stac_url"])
    rows: list[dict[str, Any]] = []
    for year in lc["years"]:
        m0, m1 = min(lc["months"]), max(lc["months"])
        end_day = 30 if m1 in (4, 6, 9, 11) else 31
        search = client.search(
            collections=[lc["collection"]],
            bbox=cfg["bbox"],
            datetime=f"{year}-{m0:02d}-01/{year}-{m1:02d}-{end_day}",
            query={"eo:cloud_cover": {"lt": lc["max_cloud_cover"]},
                   "platform": {"in": lc["platforms"]}},
        )
        for it in search.items():
            if "lwir11" not in it.assets or "qa_pixel" not in it.assets:
                continue
            rows.append({
                "scene_id": it.id,
                "scene_date": it.datetime.date().isoformat(),
                "platform": it.properties.get("platform"),
                "cloud_cover": float(it.properties.get("eo:cloud_cover", np.nan)),
                "wrs_path": str(it.properties.get("landsat:wrs_path", "")),
                "wrs_row": str(it.properties.get("landsat:wrs_row", "")),
                "st_href": it.assets["lwir11"].href,
                "qa_href": it.assets["qa_pixel"].href,
            })
    return rows


def pick_scenes(rows: list[dict[str, Any]], max_scenes: int,
                skip_tiles: list[str] | None = None) -> list[dict[str, Any]]:
    """Clearest scenes first, round-robin across WRS tiles and then years, so no single
    tile or summer dominates. Cities that straddle paths (Houston: 025 and 026) get
    scenes from each path, and silver normalizes them on their shared overlap."""
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    df["tile"] = df["wrs_path"].astype(str) + "/" + df["wrs_row"].astype(str)
    if skip_tiles:
        df = df[~df["tile"].isin(skip_tiles)]
    df["year"] = df["scene_date"].str[:4]
    df = df.sort_values("cloud_cover")
    df["r_ty"] = df.groupby(["tile", "year"]).cumcount()          # rank within tile-year
    df = df.sort_values(["r_ty", "cloud_cover"])
    df["r_t"] = df.groupby("tile").cumcount()                      # rank within tile
    df = df.sort_values(["r_t", "cloud_cover"]).head(max_scenes)
    return df.drop(columns=["year", "r_ty", "r_t"]).to_dict("records")


def read_scene(scene: dict[str, Any], bbox: list[float], scale: float, offset: float,
               decimate: int = 1) -> pd.DataFrame:
    """Read the bbox window of one scene and return one row per usable pixel."""
    import planetary_computer
    import rasterio
    from pyproj import Transformer
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MAX_RETRY": "4",
           "GDAL_HTTP_RETRY_DELAY": "2", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF"}
    with rasterio.Env(**env):
        with rasterio.open(planetary_computer.sign(scene["st_href"])) as st_ds:
            b = transform_bounds("EPSG:4326", st_ds.crs, *bbox)
            win = from_bounds(*b, transform=st_ds.transform).round_offsets().round_lengths()
            win = win.intersection(rasterio.windows.Window(0, 0, st_ds.width, st_ds.height))
            dn = st_ds.read(1, window=win)
            tr = st_ds.window_transform(win)
            crs = st_ds.crs
        with rasterio.open(planetary_computer.sign(scene["qa_href"])) as qa_ds:
            qa = qa_ds.read(1, window=win)

    if decimate > 1:  # for quick smoke tests only
        dn, qa = dn[::decimate, ::decimate], qa[::decimate, ::decimate]
        tr = tr * tr.scale(decimate, decimate)

    usable, water = qa_masks(qa)
    usable &= dn > 0
    rr, cc = np.nonzero(usable)
    if rr.size == 0:
        return pd.DataFrame(columns=PIXEL_COLUMNS)
    xs, ys = tr * (cc + 0.5, rr + 0.5)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xs, ys)
    lon, lat = np.asarray(lon), np.asarray(lat)
    inside = (lon >= bbox[0]) & (lon <= bbox[2]) & (lat >= bbox[1]) & (lat <= bbox[3])
    return pd.DataFrame({
        "scene_id": scene["scene_id"],
        "scene_date": scene["scene_date"],
        "platform": scene["platform"],
        "lon": lon[inside],
        "lat": lat[inside],
        "lst_c": dn_to_celsius(dn[rr, cc], scale, offset)[inside].astype(np.float32),
        "is_water": water[rr, cc][inside],
    })


def read_scenes_batches(batches: Iterable[pd.DataFrame], bbox: list[float], scale: float,
                        offset: float) -> Iterator[pd.DataFrame]:
    """mapInPandas body: each input row is one scene."""
    for batch in batches:
        for scene in batch.to_dict("records"):
            yield read_scene(scene, bbox, scale, offset)
