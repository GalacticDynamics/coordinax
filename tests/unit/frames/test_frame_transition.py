"""Frame transitions."""

__all__: tuple[str, ...] = ()


import pytest

import unxt as u

import coordinax.frames as cxf
import coordinax.transforms as cxfm


def test_frame_transition_returns_transform_objects() -> None:
    """Frame transitions are still built in `frames` but return transform operators."""
    op = cxf.frame_transition(cxf.alice, cxf.alex)
    assert isinstance(op, cxfm.AbstractTransform)


class TestGoingToTheNullFrameSaysSo:
    """Every route to `noframe` gives the domain error, not a dispatch error.

    The "to the null frame" rule keys on the *target*; the transformed-frame
    rule keys on the *source*. Both match a transformed frame heading for
    `noframe`, and without equal precedence plum had no ground to choose, so
    `AmbiguousLookupError` surfaced instead of the reason. Its "from the null
    frame" mirror already carried that precedence, which is why only one
    direction was affected.
    """

    R = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))

    @pytest.mark.parametrize(
        "base", [cxf.alice, cxf.noframe], ids=["ordinary-base", "null-base"]
    )
    def test_a_transformed_frame_to_noframe(self, base) -> None:
        frame = cxf.TransformedReferenceFrame(base, self.R)
        with pytest.raises(cxf.FrameTransformError, match="to the null frame"):
            cxf.frame_transition(frame, cxf.noframe)

    def test_a_plain_frame_to_noframe(self) -> None:
        """The case that always worked, kept as the control."""
        with pytest.raises(cxf.FrameTransformError, match="to the null frame"):
            cxf.frame_transition(cxf.alice, cxf.noframe)

    def test_the_other_direction_still_says_from(self) -> None:
        with pytest.raises(cxf.FrameTransformError, match="from the null frame"):
            cxf.frame_transition(cxf.noframe, cxf.alice)

    def test_null_to_null_is_still_the_identity(self) -> None:
        """The higher-precedence rule must keep winning over both refusals."""
        assert isinstance(cxf.frame_transition(cxf.noframe, cxf.noframe), cxfm.Identity)

    def test_a_transformed_frame_to_a_real_one_still_resolves(self) -> None:
        """The transformed-frame rule is not shadowed where it should apply."""
        frame = cxf.TransformedReferenceFrame(cxf.alice, self.R)
        assert isinstance(
            cxf.frame_transition(frame, cxf.alice), cxfm.AbstractTransform
        )
