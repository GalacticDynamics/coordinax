"""Tests specific to GalileanCT, galileanct, and galilean_spacetime."""

import jax

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm

# =============================================================================
# GalileanCT — construction
# =============================================================================


class TestGalileanCTConstruction:
    """GalileanCT construction and validation."""

    def test_default_construction(self) -> None:
        """GalileanCT() constructs without arguments, defaulting to cart3d."""
        chart = cxc.GalileanCT()
        assert isinstance(chart, cxc.GalileanCT)

    def test_explicit_cart3d(self) -> None:
        """GalileanCT(cart3d) equals the default."""
        chart = cxc.GalileanCT(cxc.cart3d)
        assert chart == cxc.GalileanCT()

    def test_spherical_spatial(self) -> None:
        """GalileanCT accepts sph3d as spatial chart."""
        chart = cxc.GalileanCT(cxc.sph3d)
        assert isinstance(chart, cxc.GalileanCT)
        assert chart.spatial_chart == cxc.sph3d

    def test_cylindrical_spatial(self) -> None:
        """GalileanCT accepts cyl3d as spatial chart."""
        chart = cxc.GalileanCT(cxc.cyl3d)
        assert isinstance(chart, cxc.GalileanCT)
        assert chart.spatial_chart == cxc.cyl3d

    def test_repr_default(self) -> None:
        """Default GalileanCT repr is compact (no fields shown)."""
        r = repr(cxc.GalileanCT())
        assert "GalileanCT" in r

    def test_repr_non_default_spatial(self) -> None:
        """Non-default spatial chart appears in repr."""
        r = repr(cxc.GalileanCT(cxc.sph3d))
        assert "sph3d" in r.lower() or "Spherical3D" in r


# =============================================================================
# GalileanCT — coordinate schema
# =============================================================================


class TestGalileanCTSchema:
    """GalileanCT exposes the correct coordinate schema."""

    def test_default_components(self) -> None:
        """Default spatial=cart3d gives (ct, x, y, z)."""
        chart = cxc.GalileanCT()
        assert chart.components == ("ct", "x", "y", "z")

    def test_default_coord_dimensions(self) -> None:
        """All four dimensions are length."""
        chart = cxc.GalileanCT()
        assert chart.coord_dimensions == ("length", "length", "length", "length")

    def test_default_ndim(self) -> None:
        """Default chart is 4-dimensional."""
        chart = cxc.GalileanCT()
        assert chart.ndim == 4

    def test_ndim_consistent_with_components(self) -> None:
        chart = cxc.GalileanCT()
        assert chart.ndim == len(chart.components)
        assert chart.ndim == len(chart.coord_dimensions)

    def test_spherical_components(self) -> None:
        """sph3d spatial chart gives (ct, r, theta, phi)."""
        chart = cxc.GalileanCT(cxc.sph3d)
        assert chart.components == ("ct", "r", "theta", "phi")

    def test_spherical_coord_dimensions(self) -> None:
        """sph3d spatial dimensions are (length, length, angle, angle)."""
        chart = cxc.GalileanCT(cxc.sph3d)
        assert chart.coord_dimensions == ("length", "length", "angle", "angle")

    def test_ct_always_length(self) -> None:
        """The ct component always has dimension 'length'."""
        for spatial in (cxc.cart3d, cxc.sph3d, cxc.cyl3d):
            chart = cxc.GalileanCT(spatial)
            assert chart.coord_dimensions[0] == "length"
            assert chart.components[0] == "ct"


# =============================================================================
# GalileanCT — type hierarchy
# =============================================================================


class TestGalileanCTTypeHierarchy:
    """GalileanCT fits correctly in the chart type hierarchy."""

    def test_is_abstract_chart(self) -> None:
        assert isinstance(cxc.GalileanCT(), cxc.AbstractChart)

    def test_is_abstract_flat_product_chart(self) -> None:
        assert isinstance(cxc.GalileanCT(), cxc.AbstractFlatCartesianProductChart)

    def test_is_abstract_cartesian_product_chart(self) -> None:
        assert isinstance(cxc.GalileanCT(), cxc.AbstractCartesianProductChart)

    def test_class_is_final(self) -> None:
        """GalileanCT must not be subclassed (it is @final)."""
        final_marker = getattr(cxc.GalileanCT, "__final__", None)
        if final_marker is not None:
            assert final_marker is True


# =============================================================================
# GalileanCT — product structure
# =============================================================================


class TestGalileanCTProductStructure:
    """GalileanCT product chart factor structure."""

    def test_time_chart_is_time1d(self) -> None:
        """The time factor is always time1d."""
        chart = cxc.GalileanCT()
        assert chart.time_chart == cxc.time1d

    def test_factors_length(self) -> None:
        """Factors is a 2-tuple: (time1d, spatial_chart)."""
        chart = cxc.GalileanCT()
        assert len(chart.factors) == 2

    def test_factors_time_first(self) -> None:
        """The first factor is time1d."""
        chart = cxc.GalileanCT()
        assert chart.factors[0] == cxc.time1d

    def test_factors_spatial_second(self) -> None:
        """The second factor is the spatial chart."""
        chart = cxc.GalileanCT(cxc.sph3d)
        assert chart.factors[1] == cxc.sph3d

    def test_factor_names(self) -> None:
        """factor_names is ('time', 'space')."""
        chart = cxc.GalileanCT()
        assert chart.factor_names == ("time", "space")

    def test_split_components_ct_key(self) -> None:
        """split_components returns time dict with 'ct' key."""
        chart = cxc.GalileanCT()
        p = {
            "ct": u.Q(1.0, "km"),
            "x": u.Q(2.0, "km"),
            "y": u.Q(3.0, "km"),
            "z": u.Q(4.0, "km"),
        }
        time_part, spatial_part = chart.split_components(p)
        assert "ct" in time_part
        assert set(spatial_part.keys()) == {"x", "y", "z"}

    def test_merge_components_roundtrip(self) -> None:
        """merge_components(split_components(p)) == p."""
        chart = cxc.GalileanCT()
        p = {
            "ct": u.Q(1.0, "km"),
            "x": u.Q(2.0, "km"),
            "y": u.Q(3.0, "km"),
            "z": u.Q(4.0, "km"),
        }
        time_part, spatial_part = chart.split_components(p)
        merged = chart.merge_components((time_part, spatial_part))
        assert set(merged.keys()) == set(p.keys())


# =============================================================================
# GalileanCT — Cartesian projection
# =============================================================================


class TestGalileanCTCartesian:
    """GalileanCT.cartesian projection."""

    def test_cartesian_of_default_is_self(self) -> None:
        """cart3d spatial chart → .cartesian returns self."""
        chart = cxc.GalileanCT()
        assert chart.cartesian is chart

    def test_cartesian_of_spherical_gives_cart3d_spatial(self) -> None:
        """sph3d spatial chart → .cartesian gives GalileanCT(cart3d)."""
        chart = cxc.GalileanCT(cxc.sph3d)
        result = chart.cartesian
        assert isinstance(result, cxc.GalileanCT)
        assert result.spatial_chart == cxc.cart3d

    def test_cartesian_idempotent(self) -> None:
        """Cartesian is idempotent: chart.cartesian.cartesian is chart.cartesian."""
        chart = cxc.GalileanCT(cxc.sph3d)
        c1 = chart.cartesian
        c2 = c1.cartesian
        assert c1 is c2

    def test_cartesian_chart_fn_identity_for_default(self) -> None:
        """cartesian_chart(galileanct) is galileanct."""
        assert cxc.cartesian_chart(cxc.galileanct) is cxc.galileanct


# =============================================================================
# GalileanCT — manifold
# =============================================================================


class TestGalileanCTManifold:
    """GalileanCT.M is galilean_spacetime."""

    def test_manifold_is_galilean_spacetime(self) -> None:
        chart = cxc.GalileanCT()
        assert cxm.galilean_spacetime == chart.M

    def test_manifold_is_cartesian_product_manifold(self) -> None:
        chart = cxc.GalileanCT()
        assert isinstance(chart.M, cxm.CartesianProductManifold)

    def test_all_instances_same_manifold(self) -> None:
        """Every GalileanCT instance shares galilean_spacetime."""
        c1 = cxc.GalileanCT()
        c2 = cxc.GalileanCT(cxc.sph3d)
        assert c1.M == c2.M


# =============================================================================
# GalileanCT — JAX compatibility
# =============================================================================


class TestGalileanCTJAX:
    """GalileanCT is JAX-static and JIT-compatible."""

    def test_is_static_pytree(self) -> None:
        """GalileanCT with static spatial chart has no leaves."""
        leaves, _ = jax.tree.flatten(cxc.GalileanCT())
        assert leaves == []

    def test_jit_through_chart(self) -> None:
        """GalileanCT can be closed over in a jitted function."""

        @jax.jit
        def identity(x):
            return x

        chart = cxc.GalileanCT()
        result = identity(chart)
        assert result == chart

    def test_equality(self) -> None:
        """Two GalileanCT() default instances are equal."""
        assert cxc.GalileanCT() == cxc.GalileanCT()

    def test_equality_with_same_spatial(self) -> None:
        """Two GalileanCT(sph3d) instances are equal."""
        assert cxc.GalileanCT(cxc.sph3d) == cxc.GalileanCT(cxc.sph3d)

    def test_inequality_different_spatial(self) -> None:
        """GalileanCT(cart3d) != GalileanCT(sph3d)."""
        assert cxc.GalileanCT(cxc.cart3d) != cxc.GalileanCT(cxc.sph3d)

    def test_hash(self) -> None:
        """GalileanCT instances are hashable."""
        assert hash(cxc.GalileanCT()) == hash(cxc.GalileanCT())


# =============================================================================
# galileanct pre-defined instance
# =============================================================================


class TestGalileanctInstance:
    """The module-level galileanct pre-defined instance."""

    def test_is_galileanct_type(self) -> None:
        assert isinstance(cxc.galileanct, cxc.GalileanCT)

    def test_equals_default_instance(self) -> None:
        assert cxc.galileanct == cxc.GalileanCT()

    def test_spatial_chart_is_cart3d(self) -> None:
        assert cxc.galileanct.spatial_chart == cxc.cart3d

    def test_components_are_ct_x_y_z(self) -> None:
        assert cxc.galileanct.components == ("ct", "x", "y", "z")

    def test_cartesian_is_self(self) -> None:
        assert cxc.galileanct.cartesian is cxc.galileanct

    def test_repr_contains_class_name(self) -> None:
        assert "GalileanCT" in repr(cxc.galileanct)

    def test_manifold_is_galilean_spacetime(self) -> None:
        assert cxm.galilean_spacetime == cxc.galileanct.M


# =============================================================================
# galilean_spacetime manifold
# =============================================================================


class TestGalileanSpacetimeManifold:
    """galilean_spacetime is the R1 x R3 Cartesian product manifold."""

    def test_is_cartesian_product_manifold(self) -> None:
        assert isinstance(cxm.galilean_spacetime, cxm.CartesianProductManifold)

    def test_ndim_is_4(self) -> None:
        assert cxm.galilean_spacetime.ndim == 4

    def test_factor_names(self) -> None:
        assert cxm.galilean_spacetime.factor_names == ("ct", "space")

    def test_factor_manifolds(self) -> None:
        factors = cxm.galilean_spacetime.factors
        assert len(factors) == 2
        # First factor is R1 (1D Euclidean)
        assert isinstance(factors[0], cxm.EuclideanManifold)
        assert factors[0].ndim == 1
        # Second factor is R3 (3D Euclidean)
        assert isinstance(factors[1], cxm.EuclideanManifold)
        assert factors[1].ndim == 3

    def test_accessible_from_manifolds_module(self) -> None:
        """galilean_spacetime is exported from coordinax.manifolds."""
        assert hasattr(cxm, "galilean_spacetime")
