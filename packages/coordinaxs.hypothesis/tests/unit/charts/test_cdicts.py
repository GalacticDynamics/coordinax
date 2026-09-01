"""Tests for coordinaxs-hypothesis strategies."""

import math

import jax.numpy as jnp
import pytest
import unxt as u
from hypothesis import given, settings, strategies as st

import coordinax.charts as cxc

import coordinaxs.hypothesis.main as cxst


class TestCDictStrategy:
    """Test pdict strategy for generating valid CDict objects."""

    @given(p=cxst.cdicts(cxc.cart3d))
    def test_cdict_keys_match_chart(self, p):
        """CDict keys must exactly match chart.components."""
        assert set(p.keys()) == set(cxc.cart3d.components)

    @given(p=cxst.cdicts(cxc.cart3d))
    def test_cdict_all_quantities(self, p):
        """All values must be quantity-like."""
        for v in p.values():
            assert hasattr(v, "unit") or isinstance(v, (int, float))

    @given(p=cxst.cdicts(cxc.sph3d))
    def test_cdict_mixed_dimensions(self, p):
        """Point role allows mixed dimensions from chart.coord_dimensions."""
        # Spherical has (length, angle, angle)
        assert set(p.keys()) == {"r", "theta", "phi"}
        assert u.dimension_of(p["r"]) == u.dimension("length")
        assert u.dimension_of(p["theta"]) == u.dimension("angle")
        assert u.dimension_of(p["phi"]) == u.dimension("angle")

    @given(p=cxst.cdicts(cxst.charts(filter=cxc.Abstract3D)))
    def test_cdict_with_chart_strategy(self, p):
        """Cdicts accepts chart as a strategy, drawing chart then building CDict."""
        # All 3D charts have exactly 3 components
        assert len(p) == 3
        # All keys should be strings
        assert all(isinstance(k, str) for k in p)
        # All values should be quantity-like
        assert all(hasattr(v, "unit") for v in p.values())


class TestCDictValueControl:
    """Tests showing how to control the values (e.g. quadrant) of generated CDicts."""

    @given(
        p=cxst.cdicts(
            cxc.cart3d, elements=st.floats(min_value=1, max_value=100, width=32)
        )
    )
    def test_first_octant_via_elements(self, p):
        """elements= constrains all components to positive values (first octant)."""
        assert float(p["x"].value) > 0
        assert float(p["y"].value) > 0
        assert float(p["z"].value) > 0

    @given(
        p=cxst.cdicts(
            cxc.cart3d, elements=st.floats(min_value=-100, max_value=-1, width=32)
        )
    )
    def test_negative_octant_via_elements(self, p):
        """elements= constrains all Cartesian components to negative values."""
        assert float(p["x"].value) < 0
        assert float(p["y"].value) < 0
        assert float(p["z"].value) < 0

    @given(
        p=cxst.cdicts(
            cxc.cart2d,
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, width=32),
        )
    )
    def test_bounded_range(self, p):
        """elements= with explicit bounds keeps all component magnitudes in range."""
        for key in ("x", "y"):
            val = float(p[key].value)
            assert -10 <= val <= 10

    @given(data=st.data())
    def test_second_quadrant_per_component(self, data):
        """Use st.data() to draw different element ranges per component.

        Second quadrant in 2D: x < 0, y > 0.
        """
        p_x = data.draw(
            cxst.cdicts(
                cxc.cart2d, elements=st.floats(min_value=-100, max_value=-1, width=32)
            )
        )
        p_y = data.draw(
            cxst.cdicts(
                cxc.cart2d, elements=st.floats(min_value=1, max_value=100, width=32)
            )
        )

        assert float(p_x["x"].value) < 0
        assert float(p_y["y"].value) > 0

    @given(
        p=cxst.cdicts(
            cxc.sph3d, elements=st.floats(min_value=1, max_value=100, width=32)
        )
    )
    def test_spherical_positive_elements(self, p):
        """elements= applies to all component values, including sphere angles."""
        # r, theta, phi all have positive values when elements is positive
        assert float(p["r"].value) > 0
        assert float(p["theta"].value) > 0
        assert float(p["phi"].value) > 0


@given(p=cxst.cdicts())
def test_cdicts_with_no_argument_draws_a_chart(
    p: dict[str, u.AbstractQuantity],
) -> None:
    """``cdicts()`` draws its own chart when none is given.

    The default was ``st.deferred(lambda: cxc.charts())`` -- `coordinax.charts`
    has no ``charts`` attribute, so the first draw raised `AttributeError`. It
    survived because `st.deferred` defers to draw time and the three docstring
    examples covering this path only *define* their test functions, never call
    them, so the doctests never drew from it.
    """
    assert isinstance(p, dict)
    assert p
    assert all(isinstance(v, u.AbstractQuantity) for v in p.values())


class TestMagnitudeLeavesBoundedComponentsAlone:
    """`magnitude` scales the *unbounded* coordinates and nothing else."""

    #: Under `POLAR`'s 0.05 rad lower bound, so a cap of 0.01 leaves nothing.
    TINY = (1e-3, 1e-2)

    @pytest.mark.parametrize(
        "chart",
        [
            pytest.param(cxc.sph3d, id="sph3d"),
            pytest.param(cxc.math_sph3d, id="math_sph3d"),
            pytest.param(cxc.lonlat_sph3d, id="lonlat_sph3d"),
            pytest.param(cxc.cyl3d, id="cyl3d"),
            pytest.param(cxc.polar2d, id="polar2d"),
        ],
    )
    def test_a_tiny_magnitude_is_drawable(self, chart) -> None:
        """Regression: this used to raise `InvalidArgument`, not just filter."""

        @given(p=cxst.cdicts(chart, magnitude=self.TINY))
        @settings(max_examples=5)
        def check(p) -> None:
            assert set(p) == set(chart.components)

        check()

    @given(p=cxst.cdicts(cxc.sph3d, magnitude=(1e-12, 1e-11)))
    @settings(max_examples=20)
    def test_radius_reaches_the_requested_scale(self, p) -> None:
        """An explicit floor replaces RADIAL's absolute 1e-3 m margin."""
        r = float(u.ustrip("m", p["r"]))
        assert 1e-12 <= r <= 1e-11

    @given(p=cxst.cdicts(cxc.sph3d, magnitude=(1e-12, 1e-11)))
    @settings(max_examples=20)
    def test_angles_keep_their_own_domain(self, p) -> None:
        """Theta stays a colatitude even when the length scale is 1e-12 m."""
        theta = float(u.ustrip("rad", p["theta"]))
        assert 0.0 < theta < math.pi


class TestDomainsAtAstronomicalScale:
    """A domain whose unit is far from the drawn one, which f32 cannot span."""

    CHART = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(1.0, "kpc"))

    @given(p=cxst.cdicts(CHART, dtype=jnp.float32))
    @settings(max_examples=20)
    def test_draws_in_a_unit_that_can_hold_the_bound(self, p) -> None:
        """``mu >= 1 kpc2`` is ``9.5e38 m2``, over float32's ``3.4e38`` ceiling.

        Every area unit this package draws is metre-scale, so the bound
        survives only in the domain's own unit. Drawing in a unit that cannot
        represent it would round the bound to infinity and draw nothing.
        """
        assert float(u.ustrip("kpc2", p["mu"])) >= 1.0

    @given(p=cxst.cdicts(CHART, dtype=jnp.float32))
    @settings(max_examples=20)
    def test_the_drawn_point_is_one_the_chart_accepts(self, p) -> None:
        """The point of generating from the domain at all."""
        self.CHART.check_data(p, keys=False, values=True)

    @given(p=cxst.cdicts(cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(32.0, "m"))))
    @settings(max_examples=20)
    def test_a_floor_above_the_default_cap_is_not_inverted(self, p) -> None:
        """``Delta = 32 m`` floors `mu` at 1024 m2, over the default cap 1000.

        The domain wins: capping there would ask for ``1024 <= mu <= 1000``.
        """
        assert float(u.ustrip("m2", p["mu"])) >= 1024.0
