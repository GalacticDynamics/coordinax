"""Test Coordinax Representations."""

import coordinax_hypothesis as cxst
import plum
import pytest
from hypothesis import given

import coordinax as cx


def func_has_method(f: plum.Function, sig: tuple[type, ...], /) -> bool:
    """Test if a function has a method registered for a given signature.

    Parameters
    ----------
    f : plum.Function
        The plum function to check.
    sig : tuple[type, ...]
        The signature to check for.

    Returns
    -------
    bool
        True if the function has a method registered for the given signature,
        False otherwise.

    """
    try:
        f.resolve_method(sig)
    except plum.NotFoundLookupError:
        return False
    return True


# ====================================================================


@given(rep=cxst.representations())
def test_rep_has_init_subclass(rep) -> None:
    """Test that generated representations have init subclasses."""
    assert hasattr(rep, "__init_subclass__")
    assert callable(rep.__init_subclass__)


@given(rep=cxst.representations())
def test_rep_has_components(rep) -> None:
    """Test that generated representations have components."""
    # Has components attribute
    assert hasattr(rep, "components")
    # It's a tuple of strings
    assert isinstance(rep.components, tuple)
    assert all(isinstance(comp, str) for comp in rep.components)
    # Number of components matches expected dims
    assert len(rep.components) == len(rep.coord_dimensions) == rep.dimensionality


@given(rep=cxst.representations())
def test_rep_has_coord_dimensions(rep) -> None:
    """Test that generated representations have coord_dimensions."""
    # Has coord_dimensions attribute
    assert hasattr(rep, "coord_dimensions")
    # It's a tuple of strings | None
    assert isinstance(rep.coord_dimensions, tuple)
    assert all(isinstance(comp, str | None) for comp in rep.coord_dimensions)
    # Number of dimensions matches expected dims
    assert len(rep.coord_dimensions) == len(rep.components) == rep.dimensionality


@given(rep=cxst.representations())
def test_rep_has_dimensionality(rep) -> None:
    """Test that generated representations have dimensionality."""
    # Has dimensionality attribute
    assert hasattr(rep, "dimensionality")
    # It's an int
    assert isinstance(rep.dimensionality, int)
    # Dimensionality matches lengths
    assert rep.dimensionality == len(rep.components) == len(rep.coord_dimensions)


@given(rep=cxst.representations())
def test_rep_has_cartesian(rep) -> None:
    """Test that generated representations have cartesian."""
    # Check that `cx.r.cartesian_rep` has a method registered for this type. If
    # it doesn't, skip the rest of the test.
    if not func_has_method(cx.r.cartesian_rep, (type(rep),)):
        return

    # Has cartesian attribute
    assert hasattr(rep, "cartesian")
    # It's a representation
    assert isinstance(rep.cartesian, cx.r.AbstractRep)
    # The representation is the same dimensionality
    assert rep.cartesian.dimensionality == rep.dimensionality

def test_role_derivative_chain() -> None:
    """Test derivative/antiderivative relationships between roles."""
    assert cx.r.Pos.derivative() is cx.r.Vel
    assert cx.r.Vel.derivative() is cx.r.Acc
    assert cx.r.Vel.antiderivative() is cx.r.Pos
    assert cx.r.Acc.antiderivative() is cx.r.Vel


def test_twosphere_cartesian_rep_raises() -> None:
    """TwoSphere does not have a global Cartesian 2D representation."""
    with pytest.raises(
        NotImplementedError,
        match="TwoSphere has no global Cartesian 2D representation",
    ):
        _ = cx.r.twosphere.cartesian
