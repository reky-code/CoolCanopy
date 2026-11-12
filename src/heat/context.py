"""Context layers: city boundaries (ArcGIS REST) and NLCD rasters (MRLC WCS).

Driver-side helpers. Outputs are small enough (a few million pixels) to stage as Parquet
in a UC Volume and let Spark SQL do the H3 indexing.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "heat-trees-datathon/0.1"}


# ---- boundaries ----------------------------------------------------------------------

MIN_RING_AREA_DEG2 = 1e-9   # ~0.01 m2 at Houston's latitude; anything smaller is a sliver


def _valid_ring(ring: list[list[float]]) -> bool:
    """At least 3 distinct vertices and non-negligible area (shoelace, in deg^2)."""
    pts = {(round(x, 9), round(y, 9)) for x, y, *_ in ring}
    if len(pts) < 3:
        return False
    a = 0.0
    for (x1, y1, *_), (x2, y2, *_) in zip(ring, ring[1:] + ring[:1]):
        a += x1 * y2 - x2 * y1
    return abs(a) / 2 > MIN_RING_AREA_DEG2


def _arcgis_features(layer_url: str, params: dict[str, Any], page: int = 1000
                     ) -> list[dict[str, Any]]:
    """Page through an ArcGIS query (servers cap records per request)."""
    feats: list[dict[str, Any]] = []
    offset = 0
    while True:
        p = {**params, "resultOffset": offset, "resultRecordCount": page}
        r = requests.get(f"{layer_url}/query", params=p, headers=UA, timeout=180)
        r.raise_for_status()
        fc = r.json()
        if "error" in fc:
            raise RuntimeError(f"ArcGIS error from {layer_url}: {fc['error']}")
        batch = fc.get("features", [])
        feats += batch
        more = fc.get("exceededTransferLimit") or (fc.get("properties") or {}).get(
            "exceededTransferLimit")
        if not batch or not more:
            return feats
        offset += len(batch)


def fetch_arcgis_polygons(layer_url: str, where: str, name_field: str | None,
                          simplify_deg: float = 0.0001, extra_fields: list[str] | None = None,
                          bbox4326: list[float] | None = None) -> list[dict[str, Any]]:
    """Query an ArcGIS layer as GeoJSON (EPSG:4326) and explode MultiPolygons into single
    Polygons, which is what h3_polyfillash3 handles best. Optional bbox filter + paging."""
    fields = [f for f in [name_field, *(extra_fields or [])] if f]
    params: dict[str, Any] = {"where": where, "outFields": ",".join(fields) or "*",
                              "outSR": 4326, "f": "geojson", "returnGeometry": "true",
                              "maxAllowableOffset": simplify_deg}
    if bbox4326:
        params.update({"geometry": ",".join(map(str, bbox4326)),
                       "geometryType": "esriGeometryEnvelope", "inSR": 4326,
                       "spatialRel": "esriSpatialRelIntersects"})
    out: list[dict[str, Any]] = []
    for i, feat in enumerate(_arcgis_features(layer_url, params)):
        geom = feat.get("geometry") or {}
        props = feat.get("properties") or {}
        name = str(props.get(name_field, "")) if name_field else ""
        polys = ([geom["coordinates"]] if geom.get("type") == "Polygon"
                 else geom.get("coordinates", []) if geom.get("type") == "MultiPolygon" else [])
        for j, rings in enumerate(polys):
            if not _valid_ring(rings[0] if rings else []):
                continue  # slivers collapsed by simplification break h3_polyfillash3
            rings = [rings[0]] + [r for r in rings[1:] if _valid_ring(r)]
            out.append({"feature_id": f"{i}.{j}", "name": name,
                        **{f: props.get(f) for f in (extra_fields or [])},
                        "polygon_geojson": json.dumps({"type": "Polygon", "coordinates": rings})})
    return out


# ---- NLCD rasters via WCS ------------------------------------------------------------

def fetch_wcs_geotiff(wcs_url: str, coverage_id: str, bbox4326: list[float], out_path: str,
                      native_crs: str = "EPSG:5070") -> str:
    """GetCoverage for the bbox (reprojected to the coverage's native CRS) -> GeoTIFF."""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", native_crs, always_xy=True)
    xs, ys = zip(*[t.transform(x, y) for x in (bbox4326[0], bbox4326[2])
                   for y in (bbox4326[1], bbox4326[3])])
    params = [("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"),
              ("coverageId", coverage_id), ("format", "image/geotiff"),
              ("subset", f"X({min(xs):.0f},{max(xs):.0f})"),
              ("subset", f"Y({min(ys):.0f},{max(ys):.0f})")]
    r = requests.get(wcs_url, params=params, headers=UA, timeout=600)
    r.raise_for_status()
    if not r.content[:4] in (b"II*\x00", b"MM\x00*"):
        raise RuntimeError(f"WCS did not return a GeoTIFF for {coverage_id}: {r.text[:300]}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path


def raster_to_points(path: str, bbox4326: list[float], max_valid: int = 100) -> pd.DataFrame:
    """One row per valid pixel: lon, lat, value (NLCD percent layers: 0..100)."""
    import rasterio
    from pyproj import Transformer

    with rasterio.open(path) as ds:
        a = ds.read(1)
        tr, crs = ds.transform, ds.crs
    ok = a <= max_valid
    rr, cc = np.nonzero(ok)
    xs, ys = tr * (cc + 0.5, rr + 0.5)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xs, ys)
    lon, lat = np.asarray(lon), np.asarray(lat)
    inside = ((lon >= bbox4326[0]) & (lon <= bbox4326[2]) &
              (lat >= bbox4326[1]) & (lat <= bbox4326[3]))
    return pd.DataFrame({"lon": lon[inside], "lat": lat[inside],
                         "value": a[rr, cc][inside].astype(np.uint8)})
