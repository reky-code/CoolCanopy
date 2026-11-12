import pytest

from heat import footprints as fp


def test_bbox_clause_is_overlap_test():
    c = fp.bbox_clause([-95.9, 29.5, -95.0, 30.1])
    assert "bbox.xmax >= -95.9" in c and "bbox.xmin <= -95.0" in c
    assert "bbox.ymax >= 29.5" in c and "bbox.ymin <= 30.1" in c


def test_bbox_clause_rejects_inverted_box():
    with pytest.raises(ValueError):
        fp.bbox_clause([-95.0, 29.5, -95.9, 30.1])


def test_road_buffer_sql_rail_first_and_default():
    sql = fp.road_buffer_sql({"residential": 5, "motorway": 15, "default": 4}, rail_m=6)
    assert sql.startswith("CASE WHEN subtype = 'rail' THEN 6.0")
    assert "WHEN class = 'motorway' THEN 15.0" in sql
    assert "'default'" not in sql and sql.endswith("ELSE 4.0 END")


def test_road_buffer_sql_rejects_injection():
    with pytest.raises(ValueError):
        fp.road_buffer_sql({"x' OR 1=1 --": 3}, rail_m=5)


def test_overture_queries_shape():
    cfg = {"s3_base": "s3://overturemaps-us-west-2/release/", "release": "2026-10-21.0",
           "building_setback_m": 2, "road_half_width_m": {"residential": 5}}
    q = fp.overture_queries(cfg, [-95.9, 29.5, -95.0, 30.1])
    assert set(q) == {"buildings", "roads"}
    assert "release/2026-10-21.0/theme=buildings/type=building/*" in q["buildings"]
    assert "CAST(2.0 AS DOUBLE) AS buffer_m" in q["buildings"]
    assert "subtype IN ('road', 'rail')" in q["roads"]
    for s in q.values():
        assert "ST_AsWKB(geometry) AS geom_wkb" in s


@pytest.mark.parametrize("imperv,canopy,expect", [
    (95, 0, 1),     # downtown block: ~750 m2 open -> 1 tree, not the old 28
    (20, 10, 17),   # leafy residential: lots of yard, capped below max
    (60, 45, 0),    # paved + already shaded: no room
])
def test_open_ground_capacity(imperv, canopy, expect):
    got = fp.open_ground_capacity(imperv, canopy, cell_m2=15047.5, usable_share=0.5,
                                  spot_m2=300, max_trees=30)
    assert got == expect
