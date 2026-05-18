"""Tests for Point, Tangent, and Coordinate arithmetic via Quax primitives."""

__all__: tuple[str, ...] = ()

import jax
import pytest

import quaxed.lax as qlax
import quaxed.numpy as jnp
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
        result = f(_cart_vec(1, 2, 3), _cart_vec(4, 5, 6))
        assert result["x"] == u.Q(5, "m")

    def test_cx_add(self) -> None:
        result = cxr.add(_cart_vec(1, 2, 3), _cart_vec(4, 5, 6))
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
        v = _cart_vec(4, 5, 6)
        result = v - v
        assert result["x"] == u.Q(0, "m")
        assert result["y"] == u.Q(0, "m")
        assert result["z"] == u.Q(0, "m")

    def test_jit(self) -> None:
        f = jax.jit(lambda a, b: a - b)
        result = f(_cart_vec(4, 5, 6), _cart_vec(1, 2, 3))
        assert result["x"] == u.Q(3, "m")

    def test_cx_subtract(self) -> None:
        result = cxr.subtract(_cart_vec(4, 5, 6), _cart_vec(1, 2, 3))
        assert result["x"] == u.Q(3, "m")


# ===================================================================
# Spherical add / subtract (Cartesian round-trip)
# ===================================================================


class TestSphericalAdd:
    """Spherical + Spherical uses Cartesian round-trip."""

    def test_chart_preserved(self) -> None:
        result = _sph_vec(1, 0.3, 0.5) + _sph_vec(0.5, 0.1, 0.2)
        assert result.chart == cxc.sph3d

    def test_rep_preserved(self) -> None:
        result = _sph_vec(1, 0.3, 0.5) + _sph_vec(0.5, 0.1, 0.2)
        assert result.rep == cxr.point

    def test_add_is_not_componentwise(self) -> None:
        """Sph add must NOT be component-wise (it's a Cartesian round-trip)."""
        s1 = _sph_vec(1, 0.3, 0.5)
        s2 = _sph_vec(0.5, 0.1, 0.2)
        result = s1 + s2

        # Component-wise would give r=1.5 — verify it does not.
        assert result["r"] != u.Q(1.5, "km")

    def test_jit(self) -> None:
        f = jax.jit(lambda a, b: a + b)
        result = f(_sph_vec(1, 0.3, 0.5), _sph_vec(0.5, 0.1, 0.2))
        assert result.chart == cxc.sph3d


class TestSphericalSubtract:
    """Spherical - Spherical uses Cartesian round-trip."""

    def test_chart_preserved(self) -> None:
        result = _sph_vec(1, 0.3, 0.5) - _sph_vec(0.5, 0.1, 0.2)
        assert result.chart == cxc.sph3d

    def test_rep_preserved(self) -> None:
        result = _sph_vec(1, 0.3, 0.5) - _sph_vec(0.5, 0.1, 0.2)
        assert result.rep == cxr.point

    def test_subtract_self_is_near_zero(self) -> None:
        """Sph subtract self should give r ≈ 0 (Cartesian round-trip)."""
        s = _sph_vec(1, 0.5, 0.5)
        result = s - s
        # r should be ~0 after Cartesian round-trip
        assert float(result["r"].value) == pytest.approx(0, abs=1e-6)

    def test_subtract_is_not_componentwise(self) -> None:
        """Sph subtract must NOT be component-wise."""
        s1 = _sph_vec(1, 0.3, 0.5)
        s2 = _sph_vec(0.5, 0.1, 0.2)
        result = s1 - s2

        # Component-wise would give r=0.5 — verify it does not.
        assert result["r"] != u.Q(0.5, "km")

    def test_jit(self) -> None:
        f = jax.jit(lambda a, b: a - b)
        result = f(_sph_vec(1, 0.3, 0.5), _sph_vec(0.5, 0.1, 0.2))
        assert result.chart == cxc.sph3d


# ===================================================================
# Cross-chart operations
# ===================================================================


class TestCrossChart:
    """Cross-chart arithmetic: result keeps lhs chart."""

    def test_cart_minus_sph_keeps_cart(self) -> None:
        c = _cart_vec(1, 2, 3)
        s = _sph_vec(1, 0.3, 0.5)
        result = c - s
        assert result.chart == cxc.cart3d

    def test_cart_plus_sph_keeps_cart(self) -> None:
        c = _cart_vec(1, 2, 3)
        s = _sph_vec(1, 0.3, 0.5)
        result = c + s
        assert result.chart == cxc.cart3d

    def test_sph_minus_cart_keeps_sph(self) -> None:
        c = _cart_vec(1, 2, 3)
        s = _sph_vec(1, 0.3, 0.5)
        result = s - c
        assert result.chart == cxc.sph3d

    def test_sph_plus_cart_keeps_sph(self) -> None:
        c = _cart_vec(1, 2, 3)
        s = _sph_vec(1, 0.3, 0.5)
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


# ===================================================================
# Helpers for Tangent tests
# ===================================================================


def _cart_tangent(*xyz: float, unit: str = "m/s") -> cx.Tangent:
    """Create a Cartesian coord-basis velocity Tangent."""
    return cx.Tangent.from_(
        {"x": u.Q(xyz[0], unit), "y": u.Q(xyz[1], unit), "z": u.Q(xyz[2], unit)},
        cxc.cart3d,
        cxr.coord_vel,
    )


# ===================================================================
# Tangent add
# ===================================================================


class TestTangentAdd:
    """Tangent + Tangent: component-wise (linear space, no round-trip)."""

    def test_basic(self) -> None:
        v1 = _cart_tangent(1, 2, 3)
        v2 = _cart_tangent(4, 5, 6)
        result = v1 + v2
        assert result["x"] == u.Q(5, "m/s")
        assert result["y"] == u.Q(7, "m/s")
        assert result["z"] == u.Q(9, "m/s")

    def test_chart_preserved(self) -> None:
        result = _cart_tangent(1, 2, 3) + _cart_tangent(4, 5, 6)
        assert result.chart == cxc.cart3d

    def test_rep_preserved(self) -> None:
        result = _cart_tangent(1, 2, 3) + _cart_tangent(4, 5, 6)
        assert result.rep == cxr.coord_vel

    def test_jit(self) -> None:
        f = jax.jit(lambda a, b: a + b)
        result = f(_cart_tangent(1, 2, 3), _cart_tangent(4, 5, 6))
        assert result["x"] == u.Q(5, "m/s")

    def test_cx_add(self) -> None:
        v1 = _cart_tangent(1, 2, 3)
        v2 = _cart_tangent(4, 5, 6)
        result = cxr.add(v1, v2)
        assert result["x"] == u.Q(5, "m/s")


# ===================================================================
# Tangent subtract
# ===================================================================


class TestTangentSubtract:
    """Tangent - Tangent: component-wise (linear space)."""

    def test_basic(self) -> None:
        v1 = _cart_tangent(4, 5, 6)
        v2 = _cart_tangent(1, 2, 3)
        result = v1 - v2
        assert result["x"] == u.Q(3, "m/s")
        assert result["y"] == u.Q(3, "m/s")
        assert result["z"] == u.Q(3, "m/s")

    def test_subtract_self_is_zero(self) -> None:
        v = _cart_tangent(1, 2, 3)
        result = v - v
        assert result["x"] == u.Q(0, "m/s")
        assert result["y"] == u.Q(0, "m/s")
        assert result["z"] == u.Q(0, "m/s")

    def test_chart_preserved(self) -> None:
        result = _cart_tangent(4, 5, 6) - _cart_tangent(1, 2, 3)
        assert result.chart == cxc.cart3d

    def test_rep_preserved(self) -> None:
        result = _cart_tangent(4, 5, 6) - _cart_tangent(1, 2, 3)
        assert result.rep == cxr.coord_vel

    def test_jit(self) -> None:
        f = jax.jit(lambda a, b: a - b)
        result = f(_cart_tangent(4, 5, 6), _cart_tangent(1, 2, 3))
        assert result["x"] == u.Q(3, "m/s")

    def test_cx_subtract(self) -> None:
        v1 = _cart_tangent(4, 5, 6)
        v2 = _cart_tangent(1, 2, 3)
        result = cxr.subtract(v1, v2)
        assert result["x"] == u.Q(3, "m/s")


# ===================================================================
# Tangent negation
# ===================================================================


class TestTangentNeg:
    """Unary negation of a Tangent is component-wise."""

    def test_basic(self) -> None:
        v = _cart_tangent(1, 2, 3)
        result = -v
        assert result["x"] == u.Q(-1, "m/s")
        assert result["y"] == u.Q(-2, "m/s")
        assert result["z"] == u.Q(-3, "m/s")

    def test_chart_preserved(self) -> None:
        result = -_cart_tangent(1, 2, 3)
        assert result.chart == cxc.cart3d

    def test_rep_preserved(self) -> None:
        result = -_cart_tangent(1, 2, 3)
        assert result.rep == cxr.coord_vel

    def test_double_neg_identity(self) -> None:
        v = _cart_tangent(1, 2, 3)
        neg_v = -v
        assert (-neg_v)["x"] == v["x"]

    def test_jit(self) -> None:
        f = jax.jit(lambda a: -a)
        result = f(_cart_tangent(1, 2, 3))
        assert result["x"] == u.Q(-1, "m/s")


# ===================================================================
# Tangent scalar multiplication / division
# ===================================================================


class TestTangentScalarMul:
    """Scalar * Tangent and Tangent * scalar."""

    def test_scalar_times_tangent(self) -> None:
        v = _cart_tangent(1, 2, 3)
        result = 2 * v
        assert result["x"] == u.Q(2, "m/s")
        assert result["y"] == u.Q(4, "m/s")
        assert result["z"] == u.Q(6, "m/s")

    def test_tangent_times_scalar(self) -> None:
        v = _cart_tangent(1, 2, 3)
        result = v * 3
        assert result["x"] == u.Q(3, "m/s")
        assert result["y"] == u.Q(6, "m/s")
        assert result["z"] == u.Q(9, "m/s")

    def test_chart_preserved(self) -> None:
        result = 2 * _cart_tangent(1, 2, 3)
        assert result.chart == cxc.cart3d

    def test_rep_preserved(self) -> None:
        result = 2 * _cart_tangent(1, 2, 3)
        assert result.rep == cxr.coord_vel

    def test_jit_scalar_times_tangent(self) -> None:
        f = jax.jit(lambda s, v: s * v)
        result = f(2, _cart_tangent(1, 2, 3))
        assert result["x"] == u.Q(2, "m/s")

    def test_jit_tangent_times_scalar(self) -> None:
        f = jax.jit(lambda v, s: v * s)
        result = f(_cart_tangent(1, 2, 3), 3)
        assert result["x"] == u.Q(3, "m/s")


class TestTangentScalarDiv:
    """Tangent / scalar."""

    def test_basic(self) -> None:
        v = _cart_tangent(2, 4, 6)
        result = v / 2
        assert result["x"] == u.Q(1, "m/s")
        assert result["y"] == u.Q(2, "m/s")
        assert result["z"] == u.Q(3, "m/s")

    def test_chart_preserved(self) -> None:
        result = _cart_tangent(2, 4, 6) / 2
        assert result.chart == cxc.cart3d

    def test_rep_preserved(self) -> None:
        result = _cart_tangent(2, 4, 6) / 2
        assert result.rep == cxr.coord_vel

    def test_jit(self) -> None:
        f = jax.jit(lambda v, s: v / s)
        result = f(_cart_tangent(2, 4, 6), 2)
        assert result["x"] == u.Q(1, "m/s")


# ===================================================================
# Tangent mismatched rep → error
# ===================================================================


class TestTangentMismatchedRep:
    """Adding/subtracting Tangents with different reps raises ValueError."""

    def _vel_acc(self) -> tuple:
        vel = _cart_tangent(1, 2, 3)  # coord_vel semantic
        acc = cx.Tangent.from_(
            {"x": u.Q(1, "m/s2"), "y": u.Q(0, "m/s2"), "z": u.Q(0, "m/s2")},
            cxc.cart3d,
            cxr.coord_acc,
        )
        return vel, acc

    def test_add_different_semantic_raises(self) -> None:
        vel, acc = self._vel_acc()
        with pytest.raises(ValueError, match="Cannot add Tangent vectors"):
            _ = vel + acc

    def test_sub_different_semantic_raises(self) -> None:
        vel, acc = self._vel_acc()
        with pytest.raises(ValueError, match="Cannot subtract Tangent vectors"):
            _ = vel - acc

    def test_cx_add_different_semantic_raises(self) -> None:
        vel, acc = self._vel_acc()
        with pytest.raises(ValueError, match="Cannot add Tangent vectors"):
            cxr.add(vel, acc)

    def test_cx_subtract_different_semantic_raises(self) -> None:
        vel, acc = self._vel_acc()
        with pytest.raises(ValueError, match="Cannot subtract Tangent vectors"):
            cxr.subtract(vel, acc)


# ===================================================================
# Coordinate broadcast / convert (required for JAX JIT / vmap)
# ===================================================================


class TestCoordinateBroadcast:
    """broadcast_in_dim on Coordinate must propagate to point and all tangents."""

    def _make_coord(self) -> cx.Coordinate:
        point = cx.Point.from_([1, 0, 0], "m")
        vel = cx.Tangent.from_(
            {"x": u.Q(1, "m/s"), "y": u.Q(0, "m/s"), "z": u.Q(0, "m/s")},
            cxc.cart3d,
            cxr.coord_vel,
        )
        return cx.Coordinate(point=point, velocity=vel)

    def test_broadcast_to_batch(self) -> None:
        pv = self._make_coord()
        # broadcast_to(pv, (2, 6)) — the '6' is the total component axis
        # (3 for point + 3 for velocity) so each individual component gets
        # batch shape (2,).
        result = jnp.broadcast_to(pv, (2, 6))
        assert result.point["x"].shape == (2,)
        assert result["velocity"]["x"].shape == (2,)

    def test_jit_identity(self) -> None:
        """JIT over a Coordinate requires broadcast_in_dim to succeed."""
        pv = self._make_coord()
        f = jax.jit(lambda x: x)
        result = f(pv)
        assert result.point["x"] == u.Q(1, "m")


class TestCoordinateConvertElementType:
    """convert_element_type on Coordinate must propagate to point and all tangents."""

    def _make_coord(self) -> cx.Coordinate:
        point = cx.Point.from_([1, 0, 0], "m")
        vel = cx.Tangent.from_(
            {"x": u.Q(1, "m/s"), "y": u.Q(0, "m/s"), "z": u.Q(0, "m/s")},
            cxc.cart3d,
            cxr.coord_vel,
        )
        return cx.Coordinate(point=point, velocity=vel)

    def test_convert_to_float(self) -> None:
        pv = self._make_coord()
        result = qlax.convert_element_type(pv, float)
        assert result.point["x"].dtype == float

    def test_jit_with_float_input(self) -> None:
        pv = self._make_coord()
        f = jax.jit(lambda x: x)
        result = f(pv)
        assert result.point["x"] == u.Q(1, "m")
