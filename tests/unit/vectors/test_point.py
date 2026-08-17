"""Tests for ``coordinax.vectors.Point``."""

__all__: tuple[str, ...] = ()


import jax
import pytest

import quaxed.numpy as qnp
import unxt as u

import coordinax as cx
import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.representations as cxr
import coordinax.transforms as cxfm


class TestPointFrame:
    """Tests for the ``frame`` field on ``Point``."""

    def test_default_frame_is_noframe(self):
        """Point constructed without frame defaults to noframe."""
        p = cx.Point.from_([1, 0, 0], "km")
        assert p.frame == cxf.noframe

    def test_from_point_frame_replaces_frame(self):
        """Point.from_(point, frame) replaces (not merges) an existing frame."""
        p1 = cx.Point.from_([1, 0, 0], "km", cxf.alice)
        p2 = cx.Point.from_(p1, cxf.noframe)
        assert p2.frame == cxf.noframe
        assert p2["x"] == p1["x"]

    def test_frame_preserved_after_cconvert(self):
        """Cconvert preserves the frame field."""
        p = cx.Point.from_([1, 0, 0], "km", cxf.alice)
        p_sph = p.cconvert(cxc.sph3d)
        assert p_sph.frame == cxf.alice

    def test_to_frame_returns_point(self):
        """to_frame returns a Point with the new frame."""
        rot = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
        frame = cxf.TransformedReferenceFrame(cxf.alice, rot)
        p = cx.Point.from_([1, 0, 0], "km", cxf.alice)
        p2 = p.to_frame(frame)
        assert isinstance(p2, cx.Point)
        assert p2.frame == frame

    def test_to_frame_identity_returns_self(self):
        """`to_frame` with the same (identity-transition) frame returns self."""
        p = cx.Point.from_([1, 0, 0], "km", cxf.alice)
        p2 = p.to_frame(cxf.alice)
        assert p2 is p

    def test_frame_field_auto_converts(self):
        """Frame field auto-converts via TransformedReferenceFrame.from_.

        When a non-AbstractReferenceFrame is passed.
        """
        # Passing a transform directly should be auto-converted
        p = cx.Point(
            data={"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")},
            chart=cx.cart3d,
            frame=cxf.alice,
        )
        assert isinstance(p.frame, cxf.AbstractReferenceFrame)


# Every ``Point.from_`` overload documented in ``point.py`` also accepts a
# trailing frame; each row is the leading-argument shape for one such
# overload, so that every dispatch signature is actually exercised.
_D = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
_POINT = cx.Point.from_(_D, cxc.cart3d)

POINT_FROM_WITH_FRAME_CASES = [
    pytest.param((_POINT,), id="point"),
    pytest.param((_D,), id="dict"),
    pytest.param((_D, cxc.cart3d), id="dict-chart"),
    pytest.param((_D, cxc.cart3d, cxr.point), id="dict-chart-rep"),
    pytest.param(([1.0, 2.0, 3.0], "km"), id="array-unit"),
]


@pytest.mark.parametrize("args", POINT_FROM_WITH_FRAME_CASES)
def test_point_from_with_frame(args):
    """Every ``Point.from_`` overload accepts a trailing frame."""
    p = cx.Point.from_(*args, cxf.alice)
    assert isinstance(p, cx.Point)
    assert p.frame == cxf.alice
    assert p.chart == cxc.cart3d
    assert p["x"] == u.Q(1.0, "km")
    assert p["y"] == u.Q(2.0, "km")
    assert p["z"] == u.Q(3.0, "km")


class TestPointIndexing:
    """Batch indexing (``p[i]``) respects the broadcast ``.shape`` contract."""

    def test_index_with_scalar_and_batched_components(self):
        """Scalar component broadcasts before indexing, doesn't crash (regression)."""
        p = cx.Point.from_(
            {"x": u.Q([1.0, 2.0, 3.0], "m"), "y": u.Q(5.0, "m"), "z": u.Q(0.0, "m")},
            cxc.cart3d,
        )
        assert p.shape == (3,)
        p0 = p[0]
        assert p0["x"] == u.Q(1.0, "m")
        assert p0["y"] == u.Q(5.0, "m")
        assert p0["z"] == u.Q(0.0, "m")

    def test_index_preserves_chart_and_frame(self):
        p = cx.Point.from_(
            {"x": u.Q([1.0, 2.0], "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
            cxc.cart3d,
            cxf.alice,
        )
        p0 = p[0]
        assert p0.chart == cxc.cart3d
        assert p0.frame == cxf.alice


class TestPointJAXCompat:
    """``.shape`` and ``.aval()`` work for both unitful and unitless components.

    Regression: both were briefly broken for unitless (plain-float) leaves --
    `.shape`/`.dtype` were read as attributes, which a bare Python `float`
    doesn't have.
    """

    def test_shape_unitless_components(self):
        p = cx.Point.from_({"x": 1.0, "y": 2.0, "z": 3.0}, cxc.cart3d)
        assert p.shape == ()

    def test_aval_unitless_components(self):
        p = cx.Point.from_({"x": 1.0, "y": 2.0, "z": 3.0}, cxc.cart3d)
        aval = p.aval()
        assert aval.shape == (3,)

    def test_aval_mixed_unitful_and_unitless_components(self):
        p = cx.Point.from_({"x": 0.0, "y": u.Q(2.0, "m"), "z": 0.0}, cxc.cart3d)
        aval = p.aval()
        assert aval.shape == (3,)

    def test_aval_under_eval_shape(self):
        """`jax.eval_shape` needs `.aval()`; must work for unitless components."""
        p = cx.Point.from_({"x": 1.0, "y": 2.0, "z": 3.0}, cxc.cart3d)
        result = jax.eval_shape(lambda v: v * 2, p)
        assert isinstance(result, cx.Point)


class TestPointEquality:
    """``==`` accounts for the chart and frame, and never raises."""

    def test_eq_different_chart_is_false(self):
        """Points in different charts are unequal, not a key-mismatch error."""
        p1 = cx.Point.from_([1, 2, 3], "m")
        p2 = p1.cconvert(cxc.sph3d)
        assert not bool(qnp.all(p1 == p2))

    def test_eq_different_frame_is_false(self):
        """Points with identical data but different frames are unequal."""
        p1 = cx.Point.from_([1, 2, 3], "km", cxf.alice)
        p2 = cx.Point.from_([1, 2, 3], "km", cxf.noframe)
        assert not bool(qnp.all(p1 == p2))

    def test_eq_same_chart_frame_and_data_is_true(self):
        """Identical points remain equal."""
        p1 = cx.Point.from_([1, 2, 3], "km", cxf.alice)
        p2 = cx.Point.from_([1, 2, 3], "km", cxf.alice)
        assert bool(qnp.all(p1 == p2))


class TestPointEquivalence:
    """`equivalent` is chart- and unit-invariant, but frame-strict."""

    def test_equivalent_across_charts(self):
        """The same point in different charts is equivalent (though ``!=``)."""
        p1 = cx.Point.from_([1, 2, 3], "m")
        p2 = p1.cconvert(cxc.sph3d)
        assert not bool(qnp.all(p1 == p2))  # strict equality distinguishes charts
        assert bool(qnp.all(cx.equivalent(p1, p2)))

    def test_equivalent_across_units(self):
        """The same point in different units is equivalent."""
        p1 = cx.Point.from_([1000.0, 2000.0, 3000.0], "m")
        p2 = cx.Point.from_([1.0, 2.0, 3.0], "km")
        assert bool(qnp.all(cx.equivalent(p1, p2)))

    def test_not_equivalent_different_point(self):
        """Distinct points are not equivalent."""
        p1 = cx.Point.from_([1, 2, 3], "m")
        p2 = cx.Point.from_([1, 2, 4], "m")
        assert not bool(qnp.all(cx.equivalent(p1, p2)))

    def test_equivalent_is_frame_strict(self):
        """Identical coordinates in different frames are not equivalent."""
        p1 = cx.Point.from_([1, 2, 3], "km", cxf.alice)
        p2 = cx.Point.from_([1, 2, 3], "km", cxf.noframe)
        assert not bool(qnp.all(cx.equivalent(p1, p2)))

    def test_equivalent_elementwise_over_batch(self):
        """Equivalence is evaluated element-wise over the batch."""
        p1 = cx.Point.from_([[1.0, 1, 1], [2, 2, 2]], "m")
        p2 = cx.Point.from_([[1.0, 1, 1], [9, 9, 9]], "m").cconvert(cxc.sph3d)
        result = cx.equivalent(p1, p2)
        assert bool(result[0])
        assert not bool(result[1])

    def test_equivalent_respects_tolerance(self):
        """`atol`/`rtol` control how close counts as equivalent."""
        p1 = cx.Point.from_([1.0, 0.0, 0.0], "m")
        p2 = cx.Point.from_([1.001, 0.0, 0.0], "m")
        assert not bool(qnp.all(cx.equivalent(p1, p2)))
        assert bool(qnp.all(cx.equivalent(p1, p2, atol=1e-2)))

    def test_equivalent_unitless_components(self):
        """Equivalence works for vectors with plain (unitless) array leaves."""
        p1 = cx.Point.from_({"x": 1.0, "y": 2.0, "z": 3.0}, cxc.cart3d)
        p2 = cx.Point.from_({"x": 1.0, "y": 2.0, "z": 3.0}, cxc.cart3d)
        assert bool(qnp.all(cx.equivalent(p1, p2)))
        p3 = cx.Point.from_({"x": 1.0, "y": 2.0, "z": 9.0}, cxc.cart3d)
        assert not bool(qnp.all(cx.equivalent(p1, p3)))

    def test_equivalent_unitful_vs_unitless_is_false(self):
        """A unitful and a unitless vector are not equivalent, and never raise."""
        unitful = cx.Point.from_([1.0, 2.0, 3.0], "m")
        unitless = cx.Point.from_({"x": 1.0, "y": 2.0, "z": 3.0}, cxc.cart3d)
        assert not bool(qnp.all(cx.equivalent(unitful, unitless)))
        assert not bool(qnp.all(cx.equivalent(unitless, unitful)))

    def test_equivalent_per_component_unit_mismatch_is_false(self):
        """A per-component unitful/unitless mismatch is False, and never raises."""
        # Leaves may be mixed within a vector: 'y' is unitful on one side only.
        a = cx.Point.from_({"x": 0.0, "y": 2.0, "z": 0.0}, cxc.cart3d)
        b = cx.Point.from_({"x": 0.0, "y": u.Q(2.0, "m"), "z": 0.0}, cxc.cart3d)
        assert not bool(qnp.all(cx.equivalent(a, b)))
        assert not bool(qnp.all(cx.equivalent(b, a)))

    def test_equivalent_incompatible_dimensions_is_false(self):
        """Components with incompatible dimensions are not equivalent (no raise)."""
        a = cx.Point.from_(
            {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(0.0, "m")}, cxc.cart3d
        )
        b = cx.Point.from_(
            {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "s"), "z": u.Q(0.0, "m")}, cxc.cart3d
        )
        assert not bool(qnp.all(cx.equivalent(a, b)))

    def test_equivalent_zero_component_chart_is_true(self):
        """A 0D Cartesian chart has no components: equivalence is vacuously True."""
        p = cx.Point.from_({}, cxc.cart0d, cx.point)
        assert bool(qnp.all(cx.equivalent(p, p)))

    def test_equivalent_cross_geometry_is_false(self):
        """A Point and a Tangent are never equivalent, even with matching data."""
        p = cx.Point.from_([1.0, 2.0, 3.0], "m")
        # A displacement Tangent with *matching* Cartesian components and units.
        t = cx.Tangent.from_(
            {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")},
            cxc.cart3d,
            cxr.coord_disp,
        )
        assert not bool(qnp.all(cx.equivalent(p, t)))
        assert not bool(qnp.all(cx.equivalent(t, p)))

    def test_equivalent_tangent_never_raises(self):
        """`equivalent` on non-point (Tangent) vectors returns False, never raises."""
        t = cx.Tangent.from_(
            {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")},
            cxc.cart3d,
            cxr.coord_vel,
        )
        # A Tangent cannot re-chart to Cartesian without a base point, so a naive
        # implementation would raise; the geometry guard short-circuits to False.
        assert not bool(qnp.all(cx.equivalent(t, t)))


class TestPointGeodesicDistance:
    """`geodesic_distance` measures the shortest path along the manifold."""

    def test_geodesic_distance_euclidean(self):
        """Geodesic distance is the straight-line (Cartesian) distance."""
        p = cx.Point.from_([3.0, 0.0, 0.0], "m")
        q = cx.Point.from_([0.0, 4.0, 0.0], "m")
        d = cx.geodesic_distance(p, q)
        assert isinstance(d, cx.Distance)
        assert bool(qnp.isclose(d.ustrip("m"), 5.0))

    def test_geodesic_distance_is_chart_invariant(self):
        """Geodesic distance does not depend on the chart of either operand."""
        p = cx.Point.from_([3.0, 0.0, 0.0], "m")
        q = cx.Point.from_([0.0, 4.0, 0.0], "m").cconvert(cxc.sph3d)
        assert bool(qnp.isclose(cx.geodesic_distance(p, q).ustrip("m"), 5.0))

    def test_geodesic_distance_is_unit_invariant(self):
        """Geodesic distance does not depend on the component units."""
        p = cx.Point.from_([3.0, 0.0, 0.0], "m")
        q = cx.Point.from_([0.0, 0.004, 0.0], "km")
        assert bool(qnp.isclose(cx.geodesic_distance(p, q).ustrip("m"), 5.0))

    def test_geodesic_distance_elementwise_over_batch(self):
        """Geodesic distance is evaluated element-wise over the batch."""
        p = cx.Point.from_([[3.0, 0, 0], [1, 0, 0]], "m")
        q = cx.Point.from_([[0.0, 4, 0], [0, 1, 0]], "m")
        d = cx.geodesic_distance(p, q)
        assert bool(qnp.isclose(d.ustrip("m")[0], 5.0))
        assert bool(qnp.isclose(d.ustrip("m")[1], qnp.sqrt(2.0)))

    def test_geodesic_distance_dimensionality_follows_manifold(self):
        """2-D points give a 2-D distance -- no separate ``geodesic_distance_3d``."""
        p = cx.Point.from_([3.0, 0.0], "m")
        q = cx.Point.from_([0.0, 4.0], "m")
        assert bool(qnp.isclose(cx.geodesic_distance(p, q).ustrip("m"), 5.0))

    def test_geodesic_distance_different_frames_raises(self):
        """Geodesic distance across frames is undefined without alignment."""
        p = cx.Point.from_([1.0, 0.0, 0.0], "m", cxf.alice)
        q = cx.Point.from_([0.0, 1.0, 0.0], "m", cxf.noframe)
        with pytest.raises(ValueError, match="frame"):
            cx.geodesic_distance(p, q)

    def test_geodesic_distance_different_manifolds_raises(self):
        """A 2-D and a 3-D point have no common manifold to measure on."""
        p = cx.Point.from_([3.0, 0.0], "m")
        q = cx.Point.from_([0.0, 4.0, 0.0], "m")
        with pytest.raises(ValueError, match="different manifolds"):
            cx.geodesic_distance(p, q)

    def test_geodesic_distance_unitless_components(self):
        """Geodesic distance works for vectors with plain (unitless) array leaves."""
        p = cx.Point.from_({"x": 3.0, "y": 0.0, "z": 0.0}, cxc.cart3d)
        q = cx.Point.from_({"x": 0.0, "y": 4.0, "z": 0.0}, cxc.cart3d)
        assert bool(qnp.isclose(cx.geodesic_distance(p, q), 5.0))
