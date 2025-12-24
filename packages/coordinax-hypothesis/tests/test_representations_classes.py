"""Tests for the representation_classes strategy."""

import hypothesis.strategies as st
import pytest
from coordinax_hypothesis import representation_classes
from coordinax_hypothesis._src.utils import get_all_subclasses
from hypothesis import given, settings

import coordinax as cx


@given(rep_class=representation_classes())
@settings(max_examples=50)
def test_basic_representation_class(
    rep_class: type[cx.r.AbstractRep],
) -> None:
    """Test basic representation class generation."""
    assert issubclass(rep_class, cx.r.AbstractRep)
    assert not issubclass(rep_class, type)  # Not a metaclass


@given(rep_class=representation_classes(filter=cx.r.Abstract3D))
@settings(max_examples=50)
def test_3d_representation_classes(rep_class: type[cx.r.AbstractRep]) -> None:
    """Test 3D representation class generation."""
    assert issubclass(rep_class, cx.r.Abstract3D)
    assert issubclass(rep_class, cx.r.AbstractRep)


@given(rep_class=representation_classes(filter=cx.r.Abstract1D))
@settings(max_examples=50)
def test_1d_representation_classes(rep_class: type[cx.r.AbstractRep]) -> None:
    """Test 1D representation class generation."""
    assert issubclass(rep_class, cx.r.Abstract1D)
    assert issubclass(rep_class, cx.r.AbstractRep)


@given(rep_class=representation_classes(exclude_abstract=True))
@settings(max_examples=50)
def test_concrete_classes_only(rep_class: type[cx.r.AbstractRep]) -> None:
    """Test that only concrete classes are generated when exclude_abstract=True."""
    assert issubclass(rep_class, cx.r.AbstractRep)
    # Should be a concrete class (can be instantiated)
    assert (
        not hasattr(rep_class, "__abstractmethods__")
        or not rep_class.__abstractmethods__
    )


@given(
    rep_class=representation_classes(
        filter=st.sampled_from([cx.r.Abstract1D, cx.r.Abstract2D])
    )
)
@settings(max_examples=50)
def test_dynamic_filter(rep_class: type[cx.r.AbstractRep]) -> None:
    """Test representation class generation with dynamic union class."""
    assert issubclass(rep_class, cx.r.AbstractRep)
    assert issubclass(rep_class, (cx.r.Abstract1D, cx.r.Abstract2D))


def test_warning_when_no_subclasses_found() -> None:
    """Test that a warning is raised when no subclasses match the criteria."""

    # Create a fake class that nothing will inherit from
    class _FakeClass:
        pass

    # The warning is raised when get_all_subclasses finds no matching classes
    with pytest.warns(UserWarning, match="No subclasses found"):
        result = get_all_subclasses(
            cx.r.AbstractRep,
            filter=_FakeClass,
            exclude_abstract=True,
        )

    # Should return an empty tuple
    assert result == ()
