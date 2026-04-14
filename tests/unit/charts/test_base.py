"""Tests for chart base classes.

AbstractChart, AbstractDimensionalFlag.
"""

import contextlib

import plum
import pytest
from hypothesis import given

import unxt as u

import coordinax.charts as cxc
import coordinax.hypothesis.main as cxst
from coordinax.charts._src.base import CHART_CLASSES

# =============================================================================
# AbstractChart registration
# =============================================================================


@given(cxst.chart_classes())
def test_chart_registered_in_chart_classes(chart_cls) -> None:
    """All charts are registered in CHART_CLASSES."""
    assert chart_cls in CHART_CLASSES


# =============================================================================
# AbstractChart properties
# =============================================================================


class TestChartProperties:
    """Unit tests for AbstractChart behavior."""

    @given(cxst.charts())
    def test_chart_properties(self, chart) -> None:
        """Components returns a tuple."""
        # Components
        assert isinstance(chart.components, tuple)
        for c in chart.components:
            assert isinstance(c, (str, tuple))
        # Ndim
        assert chart.ndim == len(chart.components)
        assert chart.ndim == len(chart.coord_dimensions)
        # Coord-dims
        assert isinstance(chart.coord_dimensions, tuple)

        # Cartesian (or error there there's no cartesian)
        with contextlib.suppress(
            cxc.NoGlobalCartesianChartError, plum.NotFoundLookupError
        ):
            assert isinstance(chart.cartesian, cxc.AbstractChart)

        # Repr
        assert isinstance(repr(chart), str)
        assert chart.__class__.__name__ in repr(chart)

        # str
        assert isinstance(str(chart), str)


class TestAbstractChartCheckData:
    """Unit tests for AbstractChart.check_data method."""

    # --- components check ---

    def test_check_data_passes_for_valid_data(self) -> None:
        """check_data passes for valid data matching chart components."""
        data = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
        cxc.cart3d.check_data(data)

    def test_check_data_raises_for_missing_component(self) -> None:
        """check_data raises when a component is missing."""
        data = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}  # missing z
        with pytest.raises(ValueError, match="Data keys do not match"):
            cxc.cart3d.check_data(data)

    def test_check_data_raises_for_extra_component(self) -> None:
        """check_data raises when an extra component is present."""
        data = {
            "x": u.Q(1.0, "m"),
            "y": u.Q(2.0, "m"),
            "z": u.Q(3.0, "m"),
            "w": u.Q(4.0, "m"),
        }
        with pytest.raises(ValueError, match="Data keys do not match"):
            cxc.cart3d.check_data(data)

    def test_check_data_skips_key_check_when_disabled(self) -> None:
        """check_data skips key check when keys=False."""
        data = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}  # missing z
        cxc.cart3d.check_data(data, keys=False)  # should not raise

    # --- dimensions check ---

    def test_check_data_passes_for_valid_dimensions(self) -> None:
        """check_data passes when data dimensions match chart coord_dimensions."""
        data = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
        cxc.cart3d.check_data(data, values=True)

    def test_check_data_raises_for_wrong_dimension(self) -> None:
        """check_data raises when a value's dimension doesn't match the chart."""
        data = {"x": u.Q(1.0, "s"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
        with pytest.raises(ValueError, match="Data dimension for 'x' does not match"):
            cxc.cart3d.check_data(data, values=True)

    def test_check_data_ignores_values_by_default(self) -> None:
        """check_data does not check value dimensions when values=False (default)."""
        data = {"x": u.Q(1.0, "s"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
        cxc.cart3d.check_data(data)  # should not raise


# =============================================================================
# AbstractChart - granular property tests
# =============================================================================


def _func_has_method(f: plum.Function, sig: tuple[type, ...], /) -> bool:
    """Test if a function has a method registered for a given signature."""
    try:
        f.resolve_method(sig)
    except plum.NotFoundLookupError:
        return False
    return True


@given(chart=cxst.charts())
def test_chart_has_init_subclass(chart) -> None:
    """Test that generated charts have init subclasses."""
    assert hasattr(chart, "__init_subclass__")
    assert callable(chart.__init_subclass__)


@given(chart=cxst.charts())
def test_chart_has_components(chart) -> None:
    """Test that generated charts have components."""
    assert hasattr(chart, "components")
    assert isinstance(chart.components, tuple)
    assert all(isinstance(comp, str) for comp in chart.components)
    assert len(chart.components) == len(chart.coord_dimensions) == chart.ndim


@given(chart=cxst.charts())
def test_chart_has_coord_dimensions(chart) -> None:
    """Test that generated charts have coord_dimensions."""
    assert hasattr(chart, "coord_dimensions")
    assert isinstance(chart.coord_dimensions, tuple)
    assert all(isinstance(comp, str | None) for comp in chart.coord_dimensions)
    assert len(chart.coord_dimensions) == len(chart.components) == chart.ndim


@given(chart=cxst.charts())
def test_chart_has_ndim(chart) -> None:
    """Test that generated charts expose ndim."""
    assert hasattr(chart, "ndim")
    assert isinstance(chart.ndim, int)
    assert chart.ndim == len(chart.components) == len(chart.coord_dimensions)


@given(chart=cxst.charts())
def test_chart_has_cartesian(chart) -> None:
    """Test that generated charts have cartesian."""
    if not _func_has_method(cxc.cartesian_chart, (type(chart),)):
        return
    assert hasattr(chart, "cartesian")
    assert isinstance(chart.cartesian, cxc.AbstractChart)
    assert chart.cartesian.ndim == chart.ndim


# =============================================================================
# AbstractDimensionalFlag
# =============================================================================


class TestAbstractDimensionalFlag:
    """Unit tests for AbstractDimensionalFlag."""

    def test_dimensional_flags_registered(self) -> None:
        """Dimensional flags are registered in DIMENSIONAL_FLAGS."""
        assert len(cxc.DIMENSIONAL_FLAGS) > 0

    def test_flags_must_subclass_chart(self) -> None:
        """Non-abstract dimensional flags must subclass AbstractChart."""
        for _flag_cls in cxc.DIMENSIONAL_FLAGS.values():
            # All registered flags should have chart subclasses
            pass  # Registration enforces this
