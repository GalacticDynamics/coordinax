"""Tests for chart APIs: AbstractFixedComponentsChart and cartesian_chart."""

import plum
from hypothesis import given, settings

import coordinax.charts as cxc
import coordinax_hypothesis as cxst


class TestAbstractFixedComponentsChart:
    """Behavior of charts with fixed components."""

    def test_components_are_class_level_constant(self):
        """Components remain consistent across instances."""
        cart = cxc.Cart3D()
        assert cart.components == cxc.cart3d.components

    def test_components_match_class_instance(self):
        """Test that chart instance components match those of a new instance."""
        sph = cxc.Spherical3D()
        assert sph.components == sph.__class__().components

    @given(chart_class=cxst.chart_classes(exclude_abstract=True))
    @settings(max_examples=50)
    def test_all_fixed_component_charts_have_consistent_components(
        self, chart_class: type[cxc.AbstractChart]
    ):
        """Property test: all chart instances have consistent components."""
        # Skip charts that are not fixed-component charts
        if not issubclass(chart_class, cxc.AbstractFixedComponentsChart):
            return

        # Skip charts that require parameters
        try:
            chart1 = chart_class()
            chart2 = chart_class()
        except TypeError:
            # Some charts require parameters (e.g., ProlateSpheroidal3D)
            return

        assert chart1.components == chart2.components
        assert len(chart1.components) == chart1.ndim


class TestCartesianChartFunction:
    """Tests for cartesian_chart function."""

    def test_cartesian_chart_cart3d_identity(self):
        """Test that cartesian_chart(cart3d) returns cart3d."""
        assert cxc.cartesian_chart(cxc.cart3d) is cxc.cart3d

    def test_cartesian_chart_sph3d_returns_cart3d(self):
        """Test that cartesian_chart(sph3d) returns cart3d."""
        assert cxc.cartesian_chart(cxc.sph3d) is cxc.cart3d

    def test_cartesian_chart_cyl3d_returns_cart3d(self):
        """Test that cartesian_chart(cyl3d) returns cart3d."""
        assert cxc.cartesian_chart(cxc.cyl3d) is cxc.cart3d

    def test_cartesian_chart_cart2d_identity(self):
        """Test that cartesian_chart(cart2d) returns cart2d."""
        assert cxc.cartesian_chart(cxc.cart2d) is cxc.cart2d

    def test_cartesian_chart_polar2d_returns_cart2d(self):
        """Test that cartesian_chart(polar2d) returns cart2d."""
        assert cxc.cartesian_chart(cxc.polar2d) is cxc.cart2d

    def test_cartesian_chart_is_pure(self):
        """Test that cartesian_chart is a pure function (same input → same output)."""
        result1 = cxc.cartesian_chart(cxc.sph3d)
        result2 = cxc.cartesian_chart(cxc.sph3d)
        assert result1 is result2

    @given(chart_class=cxst.chart_classes(exclude_abstract=True))
    @settings(max_examples=50)
    def test_cartesian_chart_idempotent(self, chart_class: type[cxc.AbstractChart]):
        """Property test: cartesian_chart is idempotent."""
        # Skip embedded manifolds and other special cases for now
        if issubclass(chart_class, (cxc.EmbeddedManifold,)):
            return

        # Skip charts that require parameters
        try:
            chart = chart_class()
        except TypeError:
            # Some charts require parameters (e.g., ProlateSpheroidal3D)
            return

        try:
            cart1 = cxc.cartesian_chart(chart)
            cart2 = cxc.cartesian_chart(cart1)
        except (NotImplementedError, plum.NotFoundLookupError):
            # Some charts may not have a cartesian_chart implementation yet
            # (raises NotImplementedError or NotFoundLookupError from plum)
            pass
        else:
            # Applying cartesian_chart again should give same result
            assert cart1 is cart2
