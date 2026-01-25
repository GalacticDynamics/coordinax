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
def test_chart_has_init_subclass(chart) -> None:
    """Test that generated charts have init subclasses."""
    assert hasattr(chart, "__init_subclass__")
    assert callable(chart.__init_subclass__)


@given(chart=cxst.charts())
def test_chart_has_components(chart) -> None:
    """Test that generated charts have components."""
    # Has components attribute
    assert hasattr(chart, "components")
    # It's a tuple of strings
    assert isinstance(chart.components, tuple)
    assert all(isinstance(comp, str) for comp in chart.components)
    # Number of components matches expected dims
    assert len(chart.components) == len(chart.coord_dimensions) == chart.ndim


@given(chart=cxst.charts())
def test_chart_has_coord_dimensions(chart) -> None:
    """Test that generated charts have coord_dimensions."""
    # Has coord_dimensions attribute
    assert hasattr(chart, "coord_dimensions")
    # It's a tuple of strings | None
    assert isinstance(chart.coord_dimensions, tuple)
    assert all(isinstance(comp, str | None) for comp in chart.coord_dimensions)
    # Number of dimensions matches expected dims
    assert len(chart.coord_dimensions) == len(chart.components) == chart.ndim


@given(chart=cxst.charts())
def test_chart_has_ndim(chart) -> None:
    """Test that generated charts expose ndim."""
    # Has ndim attribute
    assert hasattr(chart, "ndim")
    # It's an int
    assert isinstance(chart.ndim, int)
    # Dimension matches lengths
    assert chart.ndim == len(chart.components) == len(chart.coord_dimensions)


@given(chart=cxst.charts())
def test_chart_has_cartesian(chart) -> None:
    """Test that generated charts have cartesian."""
    # Check that `cx.charts.cartesian_chart` has a method registered for this type. If
    # it doesn't, skip the rest of the test.
    if not func_has_method(cx.charts.cartesian_chart, (type(chart),)):
        return

    # Has cartesian attribute
    assert hasattr(chart, "cartesian")
    # It's a chart
    assert isinstance(chart.cartesian, cx.charts.AbstractChart)
    # The chart is the same dimensionality
    assert chart.cartesian.ndim == chart.ndim


def test_role_derivative_chain() -> None:
    """Test derivative/antiderivative relationships between roles."""
    assert isinstance(cx.roles.phys_disp.derivative(), cx.roles.PhysVel)
    assert isinstance(cx.roles.phys_vel.derivative(), cx.roles.PhysAcc)
    assert isinstance(cx.roles.phys_vel.antiderivative(), cx.roles.PhysDisp)
    assert isinstance(cx.roles.phys_acc.antiderivative(), cx.roles.PhysVel)


def test_twosphere_cartesian_chart_raises() -> None:
    """TwoSphere does not have a global Cartesian 2D chart."""
    with pytest.raises(
        NotImplementedError,
        match="TwoSphere has no global Cartesian 2D chart",
    ):
        _ = cx.charts.twosphere.cartesian
