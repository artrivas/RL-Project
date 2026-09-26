import pytest

from src.task_discretizer import Feature, ProductDiscretizer


def test_bins_feature_with_none():
    f = Feature("k", "bins", (-1.0, 1.0), ("l", "c", "r"), none_label="none")
    assert f.n_bins == 4 and f.bin_labels == ("none", "l", "c", "r")
    assert f.bin(None) == 0
    assert f.bin(-2.0) == 1 and f.bin(-1.0) == 2 and f.bin(0.99) == 2 and f.bin(1.0) == 3


def test_angle_feature_rings_and_sides():
    # front +/-17.5, right/left up to 90, one unsplit "behind" ring
    f = Feature("t", "angle", (17.5, 90.0, 180.0), ("front", "right", "left", "behind"),
                split=(False, True, False))
    assert f.n_bins == 4
    assert f.bin(0.0) == 0 and f.bin(17.5) == 0 and f.bin(-17.5) == 0
    assert f.bin(17.6) == 1 and f.bin(-17.6) == 2 and f.bin(90.0) == 1
    assert f.bin(90.1) == 3 and f.bin(-179.0) == 3 and f.bin(540.0) == 3   # wraps to 180
    side = Feature("s", "angle", (45.0, 135.0, 180.0), ("front", "side", "behind"),
                   split=(False, False, False))
    assert side.bin(-60.0) == side.bin(60.0) == 1


def test_invalid_features_rejected():
    with pytest.raises(ValueError):
        Feature("x", "bins", (2.0, 1.0), ("a", "b", "c"))
    with pytest.raises(ValueError):
        Feature("x", "angle", (10.0, 170.0), ("a", "b", "c"), split=(False, True))
    with pytest.raises(ValueError):
        Feature("x", "bins", (1.0,), ("only one label",))


def test_product_index_roundtrip_and_labels():
    disc = ProductDiscretizer([
        Feature("a", "bins", (1.0,), ("lo", "hi")),
        Feature("b", "angle", (30.0, 180.0), ("f", "r", "l"), split=(False, True)),
    ])
    assert disc.n_states == 6
    seen = set()
    for a in (0.5, 2.0):
        for b in (0.0, 60.0, -60.0):
            s = disc({"a": a, "b": b})
            assert disc.compose(disc.decompose(s)) == s
            seen.add(s)
    assert seen == set(range(6))
    assert disc({"a": 2.0, "b": -60.0}) == disc.compose((1, 2))
    assert disc.state_to_label(disc.compose((1, 2))) == "a=hi | b=l"
