from heat.context import _valid_ring


def test_collapsed_ring_rejected():
    # Shape of Houston District A features 10.1/10.2 after ArcGIS simplification: 4 copies of a point
    p = [-95.72, 29.861]
    assert not _valid_ring([p, p, p, p])


def test_normal_ring_accepted():
    sq = [[-95.0, 29.0], [-94.99, 29.0], [-94.99, 29.01], [-95.0, 29.01], [-95.0, 29.0]]
    assert _valid_ring(sq)
