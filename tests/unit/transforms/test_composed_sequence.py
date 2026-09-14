"""`Composed` is a sequence of its transforms.

Indexing, slicing and iteration moved onto `Composed` when the one-use
`AbstractCompositeTransform` was inlined. Nothing exercised them before or
after the move -- the refactor did not break them, it revealed that they were
never covered. Slicing in particular has a branch of its own: an index returns
one transform, a slice returns another `Composed`.
"""

import pytest

import unxt as u

import coordinax.transforms as cxfm
from coordinax.transforms._src.actions.base import AbstractTransform


@pytest.fixture
def composed() -> cxfm.Composed:
    """Three distinguishable transforms, in a known order."""
    return (
        cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
        | cxfm.Reflect.from_normal([1.0, 0.0, 0.0])
        | cxfm.Identity()
    )


def test_an_index_gives_one_transform(composed) -> None:
    """The element, not a one-element `Composed`."""
    assert isinstance(composed[1], cxfm.Reflect)
    assert not isinstance(composed[1], cxfm.Composed)


def test_a_slice_gives_a_composed(composed) -> None:
    """The other `__getitem__` branch: a sub-composition, still composable."""
    part = composed[0:2]
    assert isinstance(part, cxfm.Composed)
    assert [type(op) for op in part.transforms] == [cxfm.Rotate, cxfm.Reflect]


def test_a_whole_slice_round_trips(composed) -> None:
    """Slicing everything reproduces the original's contents."""
    assert composed[:].transforms == composed.transforms


def test_iteration_yields_the_transforms_in_order(composed) -> None:
    """`__iter__` walks the tuple, so order is the composition order."""
    seen = list(composed)
    assert [type(op) for op in seen] == [cxfm.Rotate, cxfm.Reflect, cxfm.Identity]
    assert all(isinstance(op, AbstractTransform) for op in seen)


def test_iteration_and_indexing_agree(composed) -> None:
    """The two access paths are views of one tuple, not two orderings."""
    assert list(composed) == [composed[i] for i in range(len(composed.transforms))]
