"""Tests for Point, Tangent, and Coordinate arithmetic via Quax primitives."""

__all__: tuple[str, ...] = ()

import jax
import numpy as np
import plum
import pytest
from hypothesis import assume, given, strategies as st

import quaxed.lax as qlax
import quaxed.numpy as jnp
import unxt as u

import coordinax as cx
import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinaxs.hypothesis.charts as cxcst
import coordinaxs.hypothesis.representations as cxrst
import coordinaxs.hypothesis.vectors as cxvst

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
# Point / Coordinate scaling → error (gh#1025)
# ===================================================================

_scalars = st.floats(0.1, 10, allow_nan=False, allow_infinity=False)
_SCALE_OPS = (
    lambda k, x: k * x,
    lambda k, x: x * k,
    lambda k, x: x / k,
    lambda k, x: k / x,
    lambda k, x: jnp.multiply(k, x),
    lambda k, x: jnp.multiply(x, k),
    lambda k, x: jnp.divide(x, k),
    lambda k, x: jnp.divide(k, x),
)


@given(p=cxvst.vectors(), k=_scalars)
def test_scaling_a_point_raises(p: cx.Point, k: float) -> None:
    """A point is affine: ``k * p`` has no meaning, in any chart (gh#1025)."""
    for op in _SCALE_OPS:
        with pytest.raises(TypeError, match="a point is affine"):
            op(k, p)


def test_scaling_a_point_raises_under_jit() -> None:
    p = _sph_vec(1.0, 0.5, 0.5)
    with pytest.raises(TypeError, match="a point is affine"):
        jax.jit(lambda q: 2 * q)(p)


def test_scaling_a_coordinate_raises() -> None:
    """Scaling a Coordinate would scale its base point, so it is refused too."""
    c = cx.Coordinate(point=_cart_vec(1.0, 2.0, 3.0), velocity=_cart_tangent(1, 2, 3))
    for op in _SCALE_OPS:
        with pytest.raises(TypeError, match="Coordinate by a scalar"):
            op(2.0, c)

    # The fix the message suggests: scale the fibre tangents individually.
    scaled = 2.0 * c["velocity"]
    assert jnp.allclose(scaled["x"].value, 2.0 * c["velocity"]["x"].value)


def test_point_scaling_error_fixes_work() -> None:
    """Both fixes the error message suggests give the Cartesian ``2 * p``."""
    p = _sph_vec(1.0, 0.5, 0.5)
    q = p.cconvert(cxc.cart3d)
    disp = 2 * cxr.change_basis(q, cxr.coord_basis, at=q)
    pos = cx.Point.from_({n: 2 * x for n, x in q.data.items()}, q.chart)
    for out in (disp, pos):
        for n in ("x", "y", "z"):
            np.testing.assert_allclose(
                u.ustrip("km", out[n]), 2 * u.ustrip("km", q[n]), rtol=1e-12
            )


@given(
    chart=cxcst.charts(exclude=(cxc.Abstract0D, cxc.Time1D)), k=_scalars, data=st.data()
)
def test_scaling_a_tangent_is_componentwise(
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    k: float,
    data: st.DataObject,
) -> None:
    """``k * v`` scales every component, angles included, in any chart."""
    try:
        v = cx.Tangent.from_(
            data.draw(cxrst.cdicts(chart, cxr.coord_vel)), chart, cxr.coord_vel
        )
    except (TypeError, ValueError, plum.NotFoundLookupError):
        assume(False)  # noqa: FBT003
    for out, expect in ((k * v, k), (v * k, k), (v / k, 1 / k)):
        assert isinstance(out, cx.Tangent)
        assert out.chart == v.chart
        assert out.rep == v.rep
        for n in v.data:
            np.testing.assert_allclose(
                u.ustrip(v[n].unit, out[n]),
                expect * u.ustrip(v[n].unit, v[n]),
                rtol=1e-5,  # float32 draws
            )


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
# Tangent mismatched chart → error
# ===================================================================


class TestTangentMismatchedChart:
    """Adding/subtracting Tangents from different charts raises ValueError.

    Regression test: `rep` (basis + semantic) does not include the chart, so a
    guard that only checks `rep` lets through two Tangents from *different*
    manifolds whenever their charts happen to name components the same way --
    e.g. `GalileanCT` and `MinkowskiCT` both use ``(ct, x, y, z)``. Adding
    their coordinate-basis components would silently produce a value with no
    physical meaning instead of raising.
    """

    def _galilean_minkowski_vel(self) -> tuple:
        data = {
            "ct": u.Q(1.0, "m/s"),
            "x": u.Q(2.0, "m/s"),
            "y": u.Q(3.0, "m/s"),
            "z": u.Q(4.0, "m/s"),
        }
        gal = cx.Tangent.from_(data, cxc.galileanct, cxr.coord_vel)
        mink = cx.Tangent.from_(data, cxc.minkowskict, cxr.coord_vel)
        return gal, mink

    def test_same_rep_different_chart_same_keys(self) -> None:
        """Sanity check: the two charts really do collide on component names."""
        gal, mink = self._galilean_minkowski_vel()
        assert gal.rep == mink.rep
        assert gal.chart != mink.chart
        assert set(gal.data) == set(mink.data)

    def test_add_different_chart_raises(self) -> None:
        gal, mink = self._galilean_minkowski_vel()
        with pytest.raises(ValueError, match="Cannot add Tangent vectors"):
            _ = gal + mink

    def test_sub_different_chart_raises(self) -> None:
        gal, mink = self._galilean_minkowski_vel()
        with pytest.raises(ValueError, match="Cannot subtract Tangent vectors"):
            _ = gal - mink

    def test_cx_add_different_chart_raises(self) -> None:
        gal, mink = self._galilean_minkowski_vel()
        with pytest.raises(ValueError, match="Cannot add Tangent vectors"):
            cxr.add(gal, mink)

    def test_cx_subtract_different_chart_raises(self) -> None:
        gal, mink = self._galilean_minkowski_vel()
        with pytest.raises(ValueError, match="Cannot subtract Tangent vectors"):
            cxr.subtract(gal, mink)


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
