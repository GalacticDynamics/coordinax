"""Frame transitions."""

__all__: tuple[str, ...] = ()


import coordinax.frames as cxf
import coordinax.transforms as cxfm


def test_frame_transition_returns_transform_objects() -> None:
    """Frame transitions are still built in `frames` but return transform operators."""
    op = cxf.frame_transition(cxf.alice, cxf.alex)
    assert isinstance(op, cxfm.AbstractTransform)
