"""Acting a frame transition drops the frame label the data outgrew (#941)."""

__all__: tuple[str, ...] = ()


import numpy as np
import pytest

import unxt as u

import coordinax as cx
import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.representations as cxr
import coordinax.transforms as cxfm


def _alice_to_alex() -> cxfm.AbstractTransform:
    return cxf.frame_transition(cxf.alice, cxf.alex)


def _tagged_tangent(frame: cxf.AbstractReferenceFrame, /) -> cx.Tangent:
    return cx.Tangent(
        {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")},
        cxc.cart3d,
        cxr.coord_basis,
        cxr.vel,
        frame=frame,
    )


def test_act_drops_the_source_frame_label_on_a_point() -> None:
    """The operator does not know its target, so "none" is the only honest label."""
    q = cx.Point.from_([1.0, 2.0, 3.0], "m", cxf.alice)
    out = cx.act(_alice_to_alex(), None, q)
    assert isinstance(out.frame, cxf.NoFrame)


def test_act_drops_the_source_frame_label_on_a_tangent() -> None:
    out = cx.act(_alice_to_alex(), None, _tagged_tangent(cxf.alice))
    assert isinstance(out.frame, cxf.NoFrame)


def test_identity_keeps_the_label() -> None:
    """An identity moves nothing, so the label is still true."""
    q = cx.Point.from_([1.0, 2.0, 3.0], "m", cxf.alice)
    assert cx.act(cxfm.identity, None, q).frame is cxf.alice


def test_an_untagged_vector_stays_untagged() -> None:
    q = cx.Point.from_([1.0, 2.0, 3.0], "m")
    assert isinstance(cx.act(_alice_to_alex(), None, q).frame, cxf.NoFrame)


def test_the_data_is_transformed_either_way() -> None:
    """Dropping the label must not disturb the numbers."""
    q = cx.Point.from_([1.0, 2.0, 3.0], "m", cxf.alice)
    acted = cx.act(_alice_to_alex(), None, q)
    viato_frame = q.to_frame(cxf.alex)
    for k in ("x", "y", "z"):
        np.testing.assert_allclose(
            u.ustrip("m", acted.data[k]), u.ustrip("m", viato_frame.data[k]), rtol=0
        )


def test_to_frame_still_labels_its_target() -> None:
    """`to_frame` rebinds the target right after acting, so it is unaffected."""
    q = cx.Point.from_([1.0, 2.0, 3.0], "m", cxf.alice)
    assert q.to_frame(cxf.alex).frame is cxf.alex


def test_a_second_transition_can_no_longer_double_transform() -> None:
    """The footgun this closes: the old label survived and was applied again.

    An unlabelled result refuses the second hop loudly instead of silently
    rotating already-rotated data.
    """
    q = cx.Point.from_([1.0, 2.0, 3.0], "m", cxf.alice)
    out = cx.act(_alice_to_alex(), None, q)
    with pytest.raises(cxf.FrameTransformError, match="from the null frame"):
        out.to_frame(cxf.alex)


def test_a_coordinate_bundle_drops_the_label_on_every_fibre() -> None:
    crd = cx.Coordinate(
        point=cx.Point.from_([1.0, 2.0, 3.0], "m", cxf.alice),
        velocity=_tagged_tangent(cxf.alice),
    )
    out = cx.act(_alice_to_alex(), None, crd)
    assert isinstance(out.frame, cxf.NoFrame)
    assert isinstance(out["velocity"].frame, cxf.NoFrame)
