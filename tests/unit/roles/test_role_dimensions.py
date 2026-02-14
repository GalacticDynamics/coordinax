"""Unit tests for role.dimensions()."""

from hypothesis import given, settings

import unxt as u

import coordinax.charts as cxc
import coordinax.charts._src.base as cxc_base
import coordinax.roles as cxr
import coordinax_hypothesis.core as cxst

# Exclude product charts from property-based tests: their composite tuple keys
# trigger a TypeCheckError in dimensions() (known limitation).
_NON_PRODUCT_CHARTS = cxst.charts(
    exclude=(cxc_base.AbstractCartesianProductChart,),
)


class TestPointDimensions:
    """Test Point.dimensions() matches chart coordinate dimensions (order=0)."""

    def test_cart3d_dimensions(self) -> None:
        """Point dimensions for Cart3D: all length."""
        dims = cxr.Point.dimensions(cxc.cart3d)
        assert set(dims.keys()) == {"x", "y", "z"}
        for v in dims.values():
            assert v == u.dimension("length")

    def test_sph3d_dimensions(self) -> None:
        """Point dimensions for Spherical3D: length + angle + angle."""
        dims = cxr.Point.dimensions(cxc.sph3d)
        assert dims["r"] == u.dimension("length")
        assert dims["theta"] == u.dimension("angle")
        assert dims["phi"] == u.dimension("angle")

    def test_cyl3d_dimensions(self) -> None:
        """Point dimensions for Cylindrical3D: length + angle + length."""
        dims = cxr.Point.dimensions(cxc.cyl3d)
        assert dims["rho"] == u.dimension("length")
        assert dims["phi"] == u.dimension("angle")
        assert dims["z"] == u.dimension("length")


class TestPhysVelDimensions:
    """Test PhysVel.dimensions() divides by time^1."""

    def test_cart3d_velocity_dimensions(self) -> None:
        """PhysVel for Cart3D: all speed (length / time)."""
        dims = cxr.PhysVel.dimensions(cxc.cart3d)
        assert set(dims.keys()) == {"x", "y", "z"}
        for v in dims.values():
            assert v == u.dimension("speed")

    def test_sph3d_velocity_dimensions(self) -> None:
        """PhysVel for Spherical3D: speed + angular_speed + angular_speed."""
        dims = cxr.PhysVel.dimensions(cxc.sph3d)
        assert dims["r"] == u.dimension("speed")
        assert dims["theta"] == u.dimension("angular speed")
        assert dims["phi"] == u.dimension("angular speed")


class TestPhysAccDimensions:
    """Test PhysAcc.dimensions() divides by time^2."""

    def test_cart3d_acceleration_dimensions(self) -> None:
        """PhysAcc for Cart3D: all acceleration (length / time²)."""
        dims = cxr.PhysAcc.dimensions(cxc.cart3d)
        assert set(dims.keys()) == {"x", "y", "z"}
        for v in dims.values():
            assert v == u.dimension("acceleration")

    def test_sph3d_acceleration_dimensions(self) -> None:
        """PhysAcc for Spherical3D: acceleration + angular acc + angular acc."""
        dims = cxr.PhysAcc.dimensions(cxc.sph3d)
        assert dims["r"] == u.dimension("acceleration")
        assert dims["theta"] == u.dimension("angular acceleration")
        assert dims["phi"] == u.dimension("angular acceleration")


class TestPhysDispDimensions:
    """Test PhysDisp.dimensions() — same as Point dimensions (order=0)."""

    def test_cart3d_phys_disp_dimensions(self) -> None:
        """PhysDisp for Cart3D: all length (same as Point)."""
        dims = cxr.PhysDisp.dimensions(cxc.cart3d)
        assert set(dims.keys()) == {"x", "y", "z"}
        for v in dims.values():
            assert v == u.dimension("length")

    def test_sph3d_phys_disp_dimensions(self) -> None:
        """PhysDisp for Spherical3D: length + angle + angle (same as Point)."""
        dims = cxr.PhysDisp.dimensions(cxc.sph3d)
        assert dims["r"] == u.dimension("length")
        assert dims["theta"] == u.dimension("angle")
        assert dims["phi"] == u.dimension("angle")


class TestDimensionsProperty:
    """Property-based tests for role.dimensions()."""

    @given(chart=_NON_PRODUCT_CHARTS, role=cxst.roles())
    @settings(deadline=None)
    def test_dimensions_keys_match_chart_components(
        self, chart: cxc.AbstractChart, role: cxr.AbstractRole
    ) -> None:
        """dimensions() keys match the chart components."""
        dims = type(role).dimensions(chart)
        assert tuple(dims.keys()) == chart.components

    @given(chart=_NON_PRODUCT_CHARTS, role=cxst.roles())
    @settings(deadline=None)
    def test_dimensions_values_are_dimensions_or_none(
        self, chart: cxc.AbstractChart, role: cxr.AbstractRole
    ) -> None:
        """dimensions() values are AbstractDimension or None."""
        dims = type(role).dimensions(chart)
        for v in dims.values():
            assert v is None or isinstance(v, u.AbstractDimension)

    @given(chart=_NON_PRODUCT_CHARTS)
    @settings(deadline=None)
    def test_velocity_dim_is_point_dim_over_time(
        self, chart: cxc.AbstractChart
    ) -> None:
        """PhysVel dimensions = Point dimensions / time for each component."""
        pt_dims = cxr.Point.dimensions(chart)
        vel_dims = cxr.PhysVel.dimensions(chart)
        time_dim = u.dimension("time")
        for comp in chart.components:
            if pt_dims[comp] is not None:
                assert vel_dims[comp] == pt_dims[comp] / time_dim

    @given(chart=_NON_PRODUCT_CHARTS)
    @settings(deadline=None)
    def test_acceleration_dim_is_point_dim_over_time_sq(
        self, chart: cxc.AbstractChart
    ) -> None:
        """PhysAcc dimensions = Point dimensions / time² for each component."""
        pt_dims = cxr.Point.dimensions(chart)
        acc_dims = cxr.PhysAcc.dimensions(chart)
        time_dim = u.dimension("time")
        for comp in chart.components:
            if pt_dims[comp] is not None:
                assert acc_dims[comp] == pt_dims[comp] / (time_dim**2)
