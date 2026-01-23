"""Test Coordinax Representations."""

import plum
import pytest
from hypothesis import given

import coordinax as cx
import coordinax_hypothesis as cxst


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


@given(chart=cxst.charts())
def test_rep_has_init_subclass(rep) -> None:
    """Test that generated representations have init subclasses."""
    assert hasattr(rep, "__init_subclass__")
    assert callable(rep.__init_subclass__)


@given(chart=cxst.charts())
def test_rep_has_components(rep) -> None:
    """Test that generated representations have components."""
    # Has components attribute
    assert hasattr(rep, "components")
    # It's a tuple of strings
    assert isinstance(rep.components, tuple)
    assert all(isinstance(comp, str) for comp in rep.components)
    # Number of components matches expected dims
    assert len(rep.components) == len(rep.coord_dimensions) == rep.ndim


@given(chart=cxst.charts())
def test_rep_has_coord_dimensions(rep) -> None:
    """Test that generated representations have coord_dimensions."""
    # Has coord_dimensions attribute
    assert hasattr(rep, "coord_dimensions")
    # It's a tuple of strings | None
    assert isinstance(rep.coord_dimensions, tuple)
    assert all(isinstance(comp, str | None) for comp in rep.coord_dimensions)
    # Number of dimensions matches expected dims
    assert len(rep.coord_dimensions) == len(rep.components) == rep.ndim


@given(chart=cxst.charts())
def test_rep_has_ndim(rep) -> None:
    """Test that generated representations expose ndim."""
    # Has ndim attribute
    assert hasattr(rep, "ndim")
    # It's an int
    assert isinstance(rep.ndim, int)
    # Dimension matches lengths
    assert rep.ndim == len(rep.components) == len(rep.coord_dimensions)


@given(chart=cxst.charts())
def test_rep_has_cartesian(rep) -> None:
    """Test that generated representations have cartesian."""
    # Check that `cx.charts.cartesian_chart` has a method registered for this type. If
    # it doesn't, skip the rest of the test.
    if not func_has_method(cx.charts.cartesian_chart, (type(rep),)):
        return

    # Has cartesian attribute
    assert hasattr(rep, "cartesian")
    # It's a representation
    assert isinstance(rep.cartesian, cx.charts.AbstractChart)
    # The representation is the same dimensionality
    assert rep.cartesian.ndim == rep.ndim


def test_role_derivative_chain() -> None:
    """Test derivative/antiderivative relationships between roles."""
    assert isinstance(cx.roles.phys_disp.derivative(), cx.roles.PhysVel)
    assert isinstance(cx.roles.phys_vel.derivative(), cx.roles.PhysAcc)
    assert isinstance(cx.roles.phys_vel.antiderivative(), cx.roles.PhysDisp)
    assert isinstance(cx.roles.phys_acc.antiderivative(), cx.roles.PhysVel)


def test_twosphere_cartesian_chart_raises() -> None:
    """TwoSphere does not have a global Cartesian 2D representation."""
    with pytest.raises(
        NotImplementedError,
        match="TwoSphere has no global Cartesian 2D representation",
    ):
        _ = cx.charts.twosphere.cartesian
