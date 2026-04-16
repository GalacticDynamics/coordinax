"""Integration tests for pt_map and pt_map.

Key behavioural contracts verified here:

* **Round-trip identity**: transforming from chart A → B → A recovers the
  original coordinates (within numerical precision), for all registered
  concrete chart pairs.
* **Partial application equivalence**: the two partial-application overloads
  ``ptm(None, from, to)(p)`` and ``ptm(from, to)(p)`` produce the same result
  as the direct call ``ptm(p, from, to)``.
* **Chain consistency**: going A → C directly (via the general
  ``AbstractChart → Cartesian → AbstractChart`` fallback) gives the same
  result as going A → B → C when all three charts share the same ambient
  Cartesian chart.
* **realize_cartesian / unrealize_cartesian roundtrip**: composing
  ``chart.realize_cartesian`` and ``chart.unrealize_cartesian`` recovers the
  identity on the chart's domain.
"""

__all__: tuple[str, ...] = ()

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from jax.numpy import pi
from strategies import _m_qty

import unxt as u

import coordinax.charts as cxc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_pos_m = _m_qty(0.5, 10.0)
_any_m = _m_qty(-10.0, 10.0)


def _assert_cdict_approx(got, ref, *, rel=1e-5, abs=None) -> None:
    """Assert two CDicts agree component-wise, stripping each to ref's units."""
    atol = 0.0 if abs is None else abs
    for key in ref:
        got_value = np.asarray(u.ustrip(ref[key].unit, got[key]))
        ref_value = np.asarray(u.ustrip(ref[key].unit, ref[key]))
        np.testing.assert_allclose(
            got_value,
            ref_value,
            rtol=rel,
            atol=atol,
            err_msg=f"component {key!r} differs",
        )


# ---------------------------------------------------------------------------
# Partial application equivalence
# ---------------------------------------------------------------------------


class TestPartialApplicationEquivalence:
    """ptm(None, from, to)(p) and ptm(from, to)(p) agree with ptm(p, from, to)."""

    @pytest.fixture
    def cart_point(self):
        return {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}

    def test_none_partial_form(self, cart_point) -> None:
        """ptm(None, from_chart, to_chart)(p) == ptm(p, from_chart, to_chart)."""
        exp = cxc.pt_map(cart_point, cxc.cart3d, cxc.sph3d)
        got = cxc.pt_map(None, cxc.cart3d, cxc.sph3d)(cart_point)
        _assert_cdict_approx(got, exp, rel=1e-6)

    def test_charts_partial_form(self, cart_point) -> None:
        """ptm(from_chart, to_chart)(p) == ptm(p, from_chart, to_chart)."""
        exp = cxc.pt_map(cart_point, cxc.cart3d, cxc.sph3d)
        got = cxc.pt_map(cxc.cart3d, cxc.sph3d)(cart_point)
        _assert_cdict_approx(got, exp, rel=1e-6)

    def test_realization_none_partial_form(self, cart_point) -> None:
        """prm(None, from_chart, to_chart)(p) == prm(p, from_chart, to_chart)."""
        exp = cxc.pt_map(cart_point, cxc.cart3d, cxc.sph3d)
        got = cxc.pt_map(None, cxc.cart3d, cxc.sph3d)(cart_point)
        _assert_cdict_approx(got, exp, rel=1e-6)

    def test_realization_charts_partial_form(self, cart_point) -> None:
        """prm(from_chart, to_chart)(p) == prm(p, from_chart, to_chart)."""
        exp = cxc.pt_map(cart_point, cxc.cart3d, cxc.sph3d)
        got = cxc.pt_map(cxc.cart3d, cxc.sph3d)(cart_point)
        _assert_cdict_approx(got, exp, rel=1e-6)


# ---------------------------------------------------------------------------
# Round-trip: Cartesian → chart → Cartesian recovers the original point
# ---------------------------------------------------------------------------
# Going Cartesian → other → Cartesian is used (rather than other → Cartesian →
# other) because Cartesian coordinates have no domain restrictions (any real
# triple is valid).  The inverse direction can fail near poles / degenerate
# points for some charts.


def roundtrip(p, from_chart, to_chart):
    p_out = cxc.pt_map(p, from_chart, to_chart)
    return cxc.pt_map(p_out, to_chart, from_chart)


class TestCartesianRoundTrip3D:
    """cart3d → chart → cart3d ≈ identity for all 3-D non-Cartesian charts."""

    @pytest.mark.parametrize(
        "chart",
        [cxc.sph3d, cxc.cyl3d, cxc.lonlat_sph3d, cxc.loncoslat_sph3d, cxc.math_sph3d],
    )
    def test_known_point(self, chart) -> None:
        """Known concrete point round-trips correctly."""
        p = {"x": u.Q(3, "m"), "y": u.Q(4, "m"), "z": u.Q(0, "m")}
        _assert_cdict_approx(roundtrip(p, cxc.cart3d, chart), p, abs=1e-5)

    @given(x=_any_m, y=_any_m, z=_pos_m)
    @settings(deadline=None)
    def test_sph3d_roundtrip(self, x, y, z) -> None:
        """cart3d → sph3d → cart3d is identity (z > 0 avoids theta-pole issues).

        Near-polar points where sqrt(x²+y²)/z < sqrt(float32_eps) ≈ 3.5e-4 are
        excluded: arccos(z/r) rounds to 0 in float32 and x,y are irretrievably
        lost.
        """
        assume(math.hypot(float(x.value), float(y.value)) > abs(float(z.value)) * 1e-3)
        p = {"x": x, "y": y, "z": z}
        _assert_cdict_approx(roundtrip(p, cxc.cart3d, cxc.sph3d), p, abs=1e-4)

    @given(x=_any_m, y=_any_m, z=_any_m)
    @settings(deadline=None)
    def test_cyl3d_roundtrip(self, x, y, z) -> None:
        """cart3d → cyl3d → cart3d is identity."""
        p = {"x": x, "y": y, "z": z}
        _assert_cdict_approx(roundtrip(p, cxc.cart3d, cxc.cyl3d), p, abs=1e-4)


class TestCartesianRoundTrip2D:
    """cart2d → chart → cart2d ≈ identity for 2-D non-Cartesian charts."""

    @given(x=_any_m, y=_any_m)
    @settings(deadline=None)
    def test_polar2d_roundtrip(self, x, y) -> None:
        """cart2d → polar2d → cart2d is identity."""
        p = {"x": x, "y": y}
        _assert_cdict_approx(roundtrip(p, cxc.cart2d, cxc.polar2d), p, abs=1e-4)


class TestCartesianRoundTrip1D:
    """cart1d ↔ radial1d round-trips."""

    @given(x=_any_m)
    @settings(deadline=None)
    def test_cart1d_radial1d_roundtrip(self, x) -> None:
        """cart1d → radial1d → cart1d is identity."""
        p = {"x": x}
        _assert_cdict_approx(roundtrip(p, cxc.cart1d, cxc.radial1d), p, abs=1e-5)


# ---------------------------------------------------------------------------
# realize_cartesian / unrealize_cartesian roundtrip
# ---------------------------------------------------------------------------


class TestRealizeUnrealizeRoundtrip:
    """realize_cartesian followed by unrealize_cartesian is the identity."""

    @pytest.mark.parametrize(
        ("chart", "p"),
        [
            (
                cxc.sph3d,
                {
                    "r": u.Q(2, "m"),
                    "theta": u.Q(pi / 3, "rad"),
                    "phi": u.Q(pi / 4, "rad"),
                },
            ),
            (
                cxc.cyl3d,
                {"rho": u.Q(3, "m"), "phi": u.Q(pi / 6, "rad"), "z": u.Q(1, "m")},
            ),
            (cxc.polar2d, {"r": u.Q(5, "m"), "theta": u.Q(pi / 4, "rad")}),
        ],
    )
    def test_realize_unrealize_roundtrip(self, chart, p) -> None:
        """Realize → unrealize recovers original chart-space coordinates."""
        p_cart = chart.realize_cartesian(p)
        got = chart.unrealize_cartesian(p_cart)
        _assert_cdict_approx(got, p, rel=1e-5)

    def test_cartesian_realize_is_identity(self) -> None:
        """realize_cartesian on a Cartesian chart is the identity."""
        p = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
        _assert_cdict_approx(cxc.cart3d.realize_cartesian(p), p)

    def test_cartesian_unrealize_is_identity(self) -> None:
        """unrealize_cartesian on a Cartesian chart is the identity."""
        p = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
        _assert_cdict_approx(cxc.cart3d.unrealize_cartesian(p), p)


# ---------------------------------------------------------------------------
# Chain consistency: A → B → C == A → C
# ---------------------------------------------------------------------------
# The general dispatch (precedence=-1) routes A → C through A → cart(A) → C
# when no direct registration exists.  Charts sharing the same ambient
# Cartesian chart can therefore be bridged by the general path.
# Here we verify that a direct two-hop path gives the same result.


class TestChainConsistency:
    """Going A → C is consistent with A → B → C when same ambient chart."""

    def test_sph3d_to_cyl3d_matches_via_cart3d(self) -> None:
        """sph3d → cyl3d == sph3d → cart3d → cyl3d (same ambient cart3d)."""
        p_sph = {
            "r": u.Q(5, "m"),
            "theta": u.Q(pi / 2, "rad"),
            "phi": u.Q(pi / 4, "rad"),
        }
        # Direct (general dispatch via cart3d internally)
        p_cyl_direct = cxc.pt_map(p_sph, cxc.sph3d, cxc.cyl3d)

        # Explicit two-hop
        p_cart = cxc.pt_map(p_sph, cxc.sph3d, cxc.cart3d)
        p_cyl_twohop = cxc.pt_map(p_cart, cxc.cart3d, cxc.cyl3d)

        _assert_cdict_approx(p_cyl_twohop, p_cyl_direct, rel=1e-5)

    def test_lonlat_to_sph3d_matches_via_cart3d(self) -> None:
        """lonlat_sph3d → sph3d == lonlat_sph3d → cart3d → sph3d."""
        p_ll = {
            "lon": u.Q(pi / 4, "rad"),
            "lat": u.Q(pi / 6, "rad"),
            "distance": u.Q(3, "m"),
        }
        p_sph_direct = cxc.pt_map(p_ll, cxc.lonlat_sph3d, cxc.sph3d)

        p_cart = cxc.pt_map(p_ll, cxc.lonlat_sph3d, cxc.cart3d)
        p_sph_twohop = cxc.pt_map(p_cart, cxc.cart3d, cxc.sph3d)

        _assert_cdict_approx(p_sph_twohop, p_sph_direct, rel=1e-5)

    def test_identity_dispatch_returns_same_object(self) -> None:
        """ptm(p, chart, chart) returns the original dict unchanged."""
        p = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
        result = cxc.pt_map(p, cxc.cart3d, cxc.cart3d)
        assert result is p
