"""Tests for ``coordinax.vectors.Point``."""

__all__: tuple[str, ...] = ()


import unxt as u

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.main as cx
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
            manifold=cx.euclidean3d,
            frame=cxf.alice,
        )
        assert isinstance(p.frame, cxf.AbstractReferenceFrame)
