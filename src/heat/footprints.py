"""Obstacles for tree placement: building footprints and road/rail segments from Overture Maps.

Overture publishes GeoParquet on public S3. DuckDB reads only the row groups whose bbox overlaps
the city (no credentials needed), and writes one Parquet file per layer to a UC Volume. Each row
carries a buffer distance in meters so the spatial SQL step can carve "no-plant" zones out of
every H3 cell: buildings get a setback, roads get roughly their paved half-width.
"""
from __future__ import annotations

from typing import Any

DEFAULT_ROAD_HALF_WIDTH_M = 5.0


def bbox_clause(bbox: list[float]) -> str:
    """Overlap test on Overture's per-feature `bbox` struct (prunes row groups by statistics)."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if not (x0 < x1 and y0 < y1):
        raise ValueError(f"bad bbox {bbox}")
    return (f"bbox.xmax >= {x0} AND bbox.xmin <= {x1} "
            f"AND bbox.ymax >= {y0} AND bbox.ymin <= {y1}")


def road_buffer_sql(half_widths: dict[str, float], rail_m: float,
                    default_m: float = DEFAULT_ROAD_HALF_WIDTH_M) -> str:
    """CASE expression: buffer (m) per Overture segment `class`; rail by subtype."""
    whens = [f"WHEN subtype = 'rail' THEN {float(rail_m)}"]
    for cls, m in sorted(half_widths.items()):
        if cls == "default":
            continue
        if not cls.replace("_", "").isalnum():
            raise ValueError(f"bad road class {cls!r}")
        whens.append(f"WHEN class = '{cls}' THEN {float(m)}")
    default = float(half_widths.get("default", default_m))
    return "CASE " + " ".join(whens) + f" ELSE {default} END"


def overture_queries(fp: dict[str, Any], bbox: list[float]) -> dict[str, str]:
    """DuckDB SELECTs for the two obstacle layers, already in the bronze_obstacles shape:
    kind, class, buffer_m, geom_wkb (EPSG:4326)."""
    base = f"{fp['s3_base'].rstrip('/')}/{fp['release']}"
    where = bbox_clause(bbox)
    setback = float(fp.get("building_setback_m", 2.0))
    roads = road_buffer_sql(fp.get("road_half_width_m", {}), fp.get("rail_half_width_m", 5.0))
    return {
        "buildings": f"""
            SELECT 'building' AS kind, coalesce(class, subtype, 'unknown') AS class,
                   CAST({setback} AS DOUBLE) AS buffer_m, ST_AsWKB(geometry) AS geom_wkb
            FROM read_parquet('{base}/theme=buildings/type=building/*', hive_partitioning=1)
            WHERE {where}""",
        "roads": f"""
            SELECT subtype AS kind, coalesce(class, 'unknown') AS class,
                   CAST({roads} AS DOUBLE) AS buffer_m, ST_AsWKB(geometry) AS geom_wkb
            FROM read_parquet('{base}/theme=transportation/type=segment/*', hive_partitioning=1)
            WHERE {where} AND subtype IN ('road', 'rail')""",
    }


def open_ground_capacity(imperv_pct: float, canopy_pct: float, cell_m2: float,
                         usable_share: float, spot_m2: float, max_trees: int) -> int:
    """Fallback capacity when footprints are missing: trees that fit on the cell's open,
    unshaded ground (NLCD: not impervious, not canopy). Mirrors the SQL in 07."""
    open_frac = max(0.0, 1.0 - imperv_pct / 100.0 - canopy_pct / 100.0)
    return int(min(max_trees, (cell_m2 * open_frac * usable_share) // spot_m2))
