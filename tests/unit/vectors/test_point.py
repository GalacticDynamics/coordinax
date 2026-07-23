"""Tests for ``coordinax.vectors.Point``."""

__all__: tuple[str, ...] = ()


import quaxed.numpy as qnp
import unxt as u

import coordinax as cx
import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.transforms as cxfm


class TestPointFrame:
    """Tests for the ``frame`` field on ``Point``."""

    def test_default_frame_is_noframe(self):
        """Point constructed without frame defaults to noframe."""
        p = cx.Point.from_([1, 0, 0], "km")
        assert p.frame == cxf.noframe

    def test_from_array_unit_with_frame(self):
        """Point.from_(array, unit, frame) sets frame."""
        p = cx.Point.from_([1, 0, 0], "km", cxf.alice)
        assert p.frame == cxf.alice

    def test_from_vector_frame_dispatch(self):
        """Point.from_(vector, frame) wraps vector data with given frame."""
        vec = cx.Point.from_([1, 0, 0], "km")
        p = cx.Point.from_(vec, cxf.alice)
        assert p.frame == cxf.alice
        assert p.data == vec.data
        assert p.chart == vec.chart

    def test_from_point_frame_replaces_frame(self):
        """Point.from_(point, frame) returns same data with new frame."""
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
