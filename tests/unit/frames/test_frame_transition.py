"""Frame transitions."""

__all__: tuple[str, ...] = ()


import pytest

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.transforms as cxfm
import coordinax.vectors as cxv


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

    def test_noframe_to_a_transformed_frame_says_from(self) -> None:
        """The mirror of the case above, and the one branch left untested.

        A transformed *target* matches the transformed-frame rule keyed on the
        source as well as the "from the null frame" refusal, so this is the
        other half of the ambiguity the precedences settle.
        """
        frame = cxf.TransformedReferenceFrame(cxf.alice, self.R)
        with pytest.raises(cxf.FrameTransformError, match="from the null frame"):
            cxf.frame_transition(cxf.noframe, frame)

    def test_null_to_null_is_still_the_identity(self) -> None:
        """The higher-precedence rule must keep winning over both refusals."""
        assert isinstance(cxf.frame_transition(cxf.noframe, cxf.noframe), cxfm.Identity)

    def test_a_transformed_frame_to_a_real_one_still_resolves(self) -> None:
        """The transformed-frame rule is not shadowed where it should apply."""
        frame = cxf.TransformedReferenceFrame(cxf.alice, self.R)
        assert isinstance(
            cxf.frame_transition(frame, cxf.alice), cxfm.AbstractTransform
        )


class MyTransformedFrame(cxf.AbstractTransformedReferenceFrame):
    """A user's own subclass of the exported abstract transformed frame.

    Declared at module scope, not inside a test: each class object is a fresh
    plum type, and rebuilding one per test would churn the dispatch cache.
    """


class TestSubclassingTheAbstractTransformedFrame:
    """The exported abstract base, not just the `@final` concrete class.

    All three transformed-frame rules used to bind `TransformedReferenceFrame`,
    so a subclass of the documented `AbstractTransformedReferenceFrame` was a
    frame that could not transform to or from anything: `NotFoundLookupError`.
    The bodies only read `.base_frame` and `.xop`, both declared on the
    abstract class.
    """

    R = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))

    @staticmethod
    def _xyz(q: cxv.Point) -> list[float]:
        cart = q.cconvert(cxc.cart3d)
        return [float(u.ustrip("kpc", cart[k])) for k in ("x", "y", "z")]

    def test_into_the_subclass(self) -> None:
        frame = MyTransformedFrame(cxf.alice, self.R)
        op = cxf.frame_transition(cxf.alice, frame)
        got = self._xyz(op(cxv.Point.from_([1.0, 0.0, 0.0], "kpc")))
        assert jnp.allclose(jnp.asarray(got), jnp.asarray([0.0, 1.0, 0.0]), atol=1e-12)

    def test_out_of_the_subclass(self) -> None:
        frame = MyTransformedFrame(cxf.alice, self.R)
        op = cxf.frame_transition(frame, cxf.alice)
        got = self._xyz(op(cxv.Point.from_([0.0, 1.0, 0.0], "kpc")))
        assert jnp.allclose(jnp.asarray(got), jnp.asarray([1.0, 0.0, 0.0]), atol=1e-12)

    def test_the_subclass_with_itself(self) -> None:
        frame = MyTransformedFrame(cxf.alice, self.R)
        assert isinstance(cxf.frame_transition(frame, frame), cxfm.Identity)

    def test_stacked_on_the_concrete_class(self) -> None:
        """Mixing the two ends of the hierarchy still composes."""
        inner = cxf.TransformedReferenceFrame(cxf.alice, self.R)
        outer = MyTransformedFrame(inner, cxfm.Translate.from_([1, 0, 0], "kpc"))
        op = cxf.frame_transition(inner, outer)
        got = self._xyz(op(cxv.Point.from_([0.0, -1.0, 0.0], "kpc")))
        assert jnp.allclose(jnp.asarray(got), jnp.asarray([1.0, -1.0, 0.0]), atol=1e-12)


@pytest.mark.parametrize(
    ("cls", "args"),
    [
        (cxf.AbstractReferenceFrame, ()),
        (cxf.AbstractTransformedReferenceFrame, (cxf.alice, cxfm.identity)),
    ],
    ids=["AbstractReferenceFrame", "AbstractTransformedReferenceFrame"],
)
def test_the_abstract_frames_cannot_be_built(cls, args) -> None:
    """`from_` being `plum.dispatch.abstract` shaped dispatch, not construction.

    Both bases constructed happily, and an abstract instance is the cheapest
    way into the ICRS-routing recursion the astro package guards against.
    """
    with pytest.raises(TypeError, match="Cannot instantiate abstract"):
        cls(*args)


def test_the_concrete_frames_still_build() -> None:
    """The control: marking the bases abstract must not reach their subclasses."""
    assert isinstance(cxf.alice, cxf.AbstractReferenceFrame)
    frame = cxf.TransformedReferenceFrame(cxf.alice, cxfm.identity)
    assert isinstance(frame, cxf.AbstractTransformedReferenceFrame)
    assert isinstance(
        MyTransformedFrame(cxf.alice, cxfm.identity), cxf.AbstractReferenceFrame
    )
