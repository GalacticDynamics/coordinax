"""Tests for Point add/subtract arithmetic."""

__all__: tuple[str, ...] = ()

import jax
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.main as cx
import coordinax.representations as cxr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cart_vec(*xyz: float, unit: str = "m") -> cx.Point:
    return cx.Point.from_(list(xyz), unit)


def _sph_vec(r: float, theta: float, phi: float) -> cx.Point:
    return cx.Point.from_(
        {"r": u.Q(r, "km"), "theta": u.Q(theta, "rad"), "phi": u.Q(phi, "rad")},
        cxc.sph3d,
    )


# ===================================================================
# Cartesian add / subtract
# ===================================================================


class TestCartesianAdd:
    """Cartesian + Cartesian produces component-wise addition."""

    def test_basic(self) -> None:
        v1 = _cart_vec(1, 2, 3)
        v2 = _cart_vec(4, 5, 6)
        result = v1 + v2
        assert result["x"] == u.Q(5, "m")
        assert result["y"] == u.Q(7, "m")
        assert result["z"] == u.Q(9, "m")

    def test_chart_preserved(self) -> None:
        result = _cart_vec(1, 2, 3) + _cart_vec(4, 5, 6)
        assert result.chart == cxc.cart3d

    def test_rep_preserved(self) -> None:
        result = _cart_vec(1, 2, 3) + _cart_vec(4, 5, 6)
        assert result.rep == cxr.point

    def test_jit(self) -> None:
        f = jax.jit(lambda a, b: a + b)
        result = f(_cart_vec(1.0, 2.0, 3.0), _cart_vec(4.0, 5.0, 6.0))
        assert result["x"] == u.Q(5.0, "m")

    def test_cx_add(self) -> None:
        result = cx.add(_cart_vec(1, 2, 3), _cart_vec(4, 5, 6))
        assert result["x"] == u.Q(5, "m")


class TestCartesianSubtract:
    """Cartesian - Cartesian produces component-wise subtraction."""

    def test_basic(self) -> None:
        v1 = _cart_vec(4, 5, 6)
        v2 = _cart_vec(1, 2, 3)
        result = v1 - v2
        assert result["x"] == u.Q(3, "m")
        assert result["y"] == u.Q(3, "m")
        assert result["z"] == u.Q(3, "m")

    def test_chart_preserved(self) -> None:
        result = _cart_vec(4, 5, 6) - _cart_vec(1, 2, 3)
        assert result.chart == cxc.cart3d

    def test_rep_preserved(self) -> None:
        result = _cart_vec(4, 5, 6) - _cart_vec(1, 2, 3)
        assert result.rep == cxr.point

    def test_subtract_self_is_zero(self) -> None:
        v = _cart_vec(4.0, 5.0, 6.0)
        result = v - v
        assert result["x"] == u.Q(0.0, "m")
        assert result["y"] == u.Q(0.0, "m")
        assert result["z"] == u.Q(0.0, "m")

    def test_jit(self) -> None:
        f = jax.jit(lambda a, b: a - b)
        result = f(_cart_vec(4.0, 5.0, 6.0), _cart_vec(1.0, 2.0, 3.0))
        assert result["x"] == u.Q(3.0, "m")

    def test_cx_subtract(self) -> None:
        result = cx.subtract(_cart_vec(4, 5, 6), _cart_vec(1, 2, 3))
        assert result["x"] == u.Q(3, "m")


# ===================================================================
# Spherical add / subtract (Cartesian round-trip)
# ===================================================================


class TestSphericalAdd:
    """Spherical + Spherical uses Cartesian round-trip."""

    def test_chart_preserved(self) -> None:
        result = _sph_vec(1.0, 0.3, 0.5) + _sph_vec(0.5, 0.1, 0.2)
        assert result.chart == cxc.sph3d

    def test_rep_preserved(self) -> None:
        result = _sph_vec(1.0, 0.3, 0.5) + _sph_vec(0.5, 0.1, 0.2)
        assert result.rep == cxr.point

    def test_add_is_not_componentwise(self) -> None:
        """Sph add must NOT be component-wise (it's a Cartesian round-trip)."""
        s1 = _sph_vec(1.0, 0.3, 0.5)
        s2 = _sph_vec(0.5, 0.1, 0.2)
        result = s1 + s2

        # Component-wise would give r=1.5 — verify it does not.
        assert result["r"] != u.Q(1.5, "km")

    def test_jit(self) -> None:
        f = jax.jit(lambda a, b: a + b)
        result = f(_sph_vec(1.0, 0.3, 0.5), _sph_vec(0.5, 0.1, 0.2))
        assert result.chart == cxc.sph3d


class TestSphericalSubtract:
    """Spherical - Spherical uses Cartesian round-trip."""

    def test_chart_preserved(self) -> None:
        result = _sph_vec(1.0, 0.3, 0.5) - _sph_vec(0.5, 0.1, 0.2)
        assert result.chart == cxc.sph3d

    def test_rep_preserved(self) -> None:
        result = _sph_vec(1.0, 0.3, 0.5) - _sph_vec(0.5, 0.1, 0.2)
        assert result.rep == cxr.point

    def test_subtract_self_is_near_zero(self) -> None:
        """Sph subtract self should give r ≈ 0 (Cartesian round-trip)."""
        s = _sph_vec(1.0, 0.5, 0.5)
        result = s - s
        # r should be ~0 after Cartesian round-trip
        assert float(result["r"].value) == pytest.approx(0.0, abs=1e-6)

    def test_subtract_is_not_componentwise(self) -> None:
        """Sph subtract must NOT be component-wise."""
        s1 = _sph_vec(1.0, 0.3, 0.5)
        s2 = _sph_vec(0.5, 0.1, 0.2)
        result = s1 - s2

        # Component-wise would give r=0.5 — verify it does not.
        assert result["r"] != u.Q(0.5, "km")

    def test_jit(self) -> None:
        f = jax.jit(lambda a, b: a - b)
        result = f(_sph_vec(1.0, 0.3, 0.5), _sph_vec(0.5, 0.1, 0.2))
        assert result.chart == cxc.sph3d


# ===================================================================
# Cross-chart operations
# ===================================================================


class TestCrossChart:
    """Cross-chart arithmetic: result keeps lhs chart."""

    def test_cart_minus_sph_keeps_cart(self) -> None:
        c = _cart_vec(1.0, 2.0, 3.0)
        s = _sph_vec(1.0, 0.3, 0.5)
        result = c - s
        assert result.chart == cxc.cart3d

    def test_cart_plus_sph_keeps_cart(self) -> None:
        c = _cart_vec(1.0, 2.0, 3.0)
        s = _sph_vec(1.0, 0.3, 0.5)
        result = c + s
        assert result.chart == cxc.cart3d

    def test_sph_minus_cart_keeps_sph(self) -> None:
        c = _cart_vec(1.0, 2.0, 3.0)
        s = _sph_vec(1.0, 0.3, 0.5)
        result = s - c
        assert result.chart == cxc.sph3d

    def test_sph_plus_cart_keeps_sph(self) -> None:
        c = _cart_vec(1.0, 2.0, 3.0)
        s = _sph_vec(1.0, 0.3, 0.5)
        result = s + c
        assert result.chart == cxc.sph3d


# ===================================================================
# Mismatched rep → error
# ===================================================================


class TestMismatchedRep:
    """Point with different reps cannot be added/subtracted."""

    @pytest.mark.skip(reason="Only one rep type (point) currently exists")
    def test_different_rep_raises(self) -> None:
        pass
