"""Tests for Point, Tangent, and Coordinate arithmetic via Quax primitives."""

__all__: tuple[str, ...] = ()

import operator
from dataclasses import replace

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
import coordinax.frames as cxf
import coordinax.representations as cxr
import coordinax.transforms as cxfm
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
# Point arithmetic: the affine operations
# ===================================================================

# 3-D Euclidean charts. Every point drawn below stays away from the z-axis,
# where the curvilinear charts are singular.
_E3_CHARTS = (
    cxc.cart3d,
    cxc.sph3d,
    cxc.cyl3d,
    cxc.lonlat_sph3d,
    cxc.loncoslat_sph3d,
    cxc.math_sph3d,
)
_charts = st.sampled_from(_E3_CHARTS)
_coord = st.floats(-10, 10, allow_nan=False, allow_infinity=False)


@st.composite
def _xyz(draw: st.DrawFn) -> tuple[float, float, float]:
    x, y, z = draw(_coord), draw(_coord), draw(_coord)
    assume(x * x + y * y > 0.01)
    return x, y, z


def _pt(xyz, chart=cxc.cart3d, frame=None) -> cx.Point:
    p = cx.Point.from_(list(xyz), "km")
    if frame is not None:
        p = cx.Point.from_(p, frame)
    return p.cconvert(chart)


def _dpl(*xyz: float, unit: str = "km") -> cx.Tangent:
    return cx.Tangent.from_(u.Q(list(xyz), unit), cxc.cart3d, cxr.coord_basis, cxr.dpl)


def _km(t: cx.Tangent) -> np.ndarray:
    """Cartesian components of a Cartesian tangent, in km."""
    assert t.chart == cxc.cart3d
    return np.array([float(u.ustrip("km", t[k])) for k in ("x", "y", "z")])


def _phys_basis_broken(chart, basis) -> bool:
    # TODO: the physical basis of `loncoslat_sph3d` fails with a unit error
    # (`km(3/2) / rad(1/2)`) on main too; drop this once that is fixed.
    return chart == cxc.loncoslat_sph3d and basis == cxr.phys_basis


def _close(a: cx.Point, b: cx.Point) -> bool:
    return bool(cx.equivalent(a, b, atol=1e-8, rtol=1e-8))


class TestPointDifference:
    """``p - q`` is the displacement from ``q`` to ``p``."""

    def test_is_a_cartesian_displacement(self) -> None:
        p = _sph_vec(1, 0.3, 0.5)
        q = _cart_vec(1, 2, 3, unit="km")
        d = p - q
        assert isinstance(d, cx.Tangent)
        assert d.semantic == cxr.dpl
        assert d.chart == cxc.cart3d
        assert d.basis == cxr.coord_basis
        assert d.frame == p.frame

    def test_components(self) -> None:
        d = _cart_vec(4, 5, 6) - _cart_vec(1, 2, 3)
        assert d == _dpl(3, 3, 3, unit="m")

    def test_is_not_componentwise_in_a_curvilinear_chart(self) -> None:
        s = _sph_vec(1, 0.5, 0.5)
        d = s - s
        np.testing.assert_allclose(_km(d), 0.0, atol=1e-12)

    def test_cx_subtract_and_jnp_subtract(self) -> None:
        p, q = _cart_vec(4, 5, 6), _cart_vec(1, 2, 3)
        assert cxr.subtract(p, q) == p - q
        assert jnp.subtract(p, q) == p - q

    def test_jit(self) -> None:
        p, q = _sph_vec(1, 0.3, 0.5), _sph_vec(0.5, 0.1, 0.2)
        out = jax.jit(lambda a, b: a - b)(p, q)
        np.testing.assert_allclose(_km(out), _km(p - q), rtol=1e-12)

    def test_batch_minus_scalar_broadcasts(self) -> None:
        ps = cx.Point.from_(u.Q([[1.0, 2, 3], [4, 5, 6]], "km"))
        q = _cart_vec(1, 1, 1, unit="km")
        d = ps - q
        assert d.shape == (2,)
        np.testing.assert_allclose(u.ustrip("km", d["x"]), [0.0, 3.0])

    def test_vmap_matches_eager(self) -> None:
        ps = cx.Point.from_(u.Q([[1.0, 2, 3], [4, 5, 6]], "km")).cconvert(cxc.sph3d)
        qs = cx.Point.from_(u.Q([[0.0, 1, 1], [2, 2, 2]], "km"))
        out = jax.vmap(lambda a, b: a - b)(ps, qs)
        np.testing.assert_allclose(
            u.ustrip("km", out["y"]), u.ustrip("km", (ps - qs)["y"]), rtol=1e-12
        )


class TestPointTranslation:
    """``p + d``, ``d + p`` and ``p - d`` translate ``p`` by a displacement."""

    def test_add_keeps_the_point_chart(self) -> None:
        p = _sph_vec(1, 0.3, 0.5)
        out = p + _dpl(1, 0, 0)
        assert isinstance(out, cx.Point)
        assert out.chart == cxc.sph3d

    def test_add_sub_and_reflected(self) -> None:
        p = _cart_vec(1, 2, 3, unit="km")
        d = _dpl(1, 1, 1)
        assert p + d == _cart_vec(2, 3, 4, unit="km")
        assert d + p == p + d
        assert p - d == _cart_vec(0, 1, 2, unit="km")
        assert cxr.add(p, d) == p + d
        assert jnp.add(p, d) == p + d
        assert jnp.subtract(p, d) == p - d

    def test_jit(self) -> None:
        p, d = _sph_vec(1, 0.3, 0.5), _dpl(1, 1, 1)
        assert _close(jax.jit(lambda a, b: a + b)(p, d), p + d)


class TestPointRefusals:
    """The origin-dependent operations raise, with an explanation."""

    p = _cart_vec(1, 2, 3)

    @pytest.mark.parametrize(
        "f",
        [
            lambda p: p + p,
            lambda p: -p,
            lambda p: (p - p) - p,
            lambda p: cxr.add(p, p),
            lambda p: jnp.add(p, p),
            lambda p: jnp.negative(p),
        ],
        ids=["p+q", "-p", "d-p", "cxr.add", "jnp.add", "jnp.negative"],
    )
    def test_needs_an_origin(self, f) -> None:
        with pytest.raises(TypeError, match="a point is affine"):
            f(self.p)

    def test_needs_an_origin_under_jit(self) -> None:
        with pytest.raises(TypeError, match="a point is affine"):
            jax.jit(lambda a: a + a)(self.p)

    @pytest.mark.parametrize("semantic", [cxr.vel, cxr.acc])
    def test_only_a_displacement_translates(self, semantic) -> None:
        unit = {cxr.vel: "m/s", cxr.acc: "m/s2"}[semantic]
        v = cx.Tangent.from_(
            u.Q([1.0, 0, 0], unit), cxc.cart3d, cxr.coord_basis, semantic
        )
        for f in (lambda: self.p + v, lambda: v + self.p, lambda: self.p - v):
            with pytest.raises(TypeError, match="only a displacement"):
                f()

    def test_non_affine_manifold(self) -> None:
        s = cx.Point.from_(
            {"theta": u.Angle(1.0, "rad"), "phi": u.Angle(0.5, "rad")}, cxc.sph2
        )
        with pytest.raises(TypeError, match="affine space"):
            _ = s - s
        with pytest.raises(TypeError, match="a point is affine"):
            _ = s + s
        with pytest.raises(TypeError, match="a point is affine"):
            _ = -s

    def test_different_manifolds(self) -> None:
        m = cx.Point.from_(u.Q([0.0, 1.0, 2.0, 3.0], "km"), cxc.minkowskict)
        with pytest.raises(TypeError, match="different manifolds"):
            _cart_vec(1, 2, 3, unit="km") - m
        d = cx.Tangent.from_(
            u.Q([0.0, 1.0, 2.0, 3.0], "km"), cxc.minkowskict, cxr.coord_disp
        )
        with pytest.raises(TypeError, match="different manifold"):
            _cart_vec(1, 2, 3, unit="km") + d

    def test_non_vector_operand_is_unsupported(self) -> None:
        with pytest.raises(TypeError, match="unsupported operand"):
            _ = self.p + 1

    @pytest.mark.parametrize(
        "f",
        [
            lambda c: c - c,
            lambda c: c + c,
            lambda c: -c,
            lambda c: c + c.point,
            lambda c: c.point + c,
            lambda c: c.point - c,
            jnp.negative,
            lambda c: jnp.add(c, c),
            lambda c: jnp.subtract(c, c),
        ],
        ids=["c-c", "c+c", "-c", "c+p", "p+c", "p-c", "jnp.neg", "jnp.add", "jnp.sub"],
    )
    def test_coordinate(self, f) -> None:
        c = cx.Coordinate(point=self.p, velocity=_cart_tangent(1, 2, 3))
        with pytest.raises(TypeError, match="Coordinate"):
            f(c)


class TestPointFrames:
    """The operators need one frame; ``add``/``subtract`` transform first."""

    @staticmethod
    def _astro():
        import coordinaxs.astro as cxastro

        return cxastro

    def test_operators_refuse_a_frame_mismatch(self) -> None:
        cxastro = self._astro()
        g = _pt((1, 2, 3), frame=cxastro.Galactocentric())
        i = _pt((1, 2, 3), frame=cxastro.ICRS())
        for f in (lambda: g - i, lambda: g + (i - i), lambda: g - (i - i)):
            with pytest.raises(ValueError, match=r"p\.(subtract|add)\(other, tau"):
                f()
        with pytest.raises(ValueError, match="tau"):
            jax.jit(lambda a, b: a - b)(g, i)

    def test_same_parametrised_frame_under_jit(self) -> None:
        g = _pt((1, 2, 3), frame=self._astro().Galactocentric())
        out = jax.jit(lambda a, b: a - b)(g, g)
        np.testing.assert_allclose(_km(out), 0.0)
        assert type(out.frame) is type(g.frame)

    def test_subtract_across_a_static_frame(self) -> None:
        p = _pt((10, 0, 0), frame=cxf.alex)
        q = _pt((1, 2, 3), frame=cxf.alice)
        out = p.subtract(q)
        assert out.frame == cxf.alex
        np.testing.assert_allclose(_km(out), _km(p - q.to_frame(cxf.alex)))
        assert p.subtract(p.to_frame(cxf.alex)) == p - p

    def test_add_across_a_static_frame(self) -> None:
        p = _pt((0, 0, 0), frame=cxf.alex)
        d = replace(_dpl(1, 0, 0), frame=cxf.alice)
        out = p.add(d)
        assert out.frame == cxf.alex
        assert _close(out, p + replace(_dpl(0, 1, 0), frame=cxf.alex))

    @given(a=_xyz(), b=_xyz(), t=st.floats(-5, 5, allow_nan=False))
    def test_subtract_across_a_time_dependent_frame(self, a, b, t) -> None:
        """``p.subtract(q, tau=t)`` is transform-then-subtract at ``t``."""
        tau = u.Q(t, "s")
        p = _pt(a, frame=SPINNING)
        q = _pt(b, frame=cxf.alice)
        op = cxf.frame_transition(cxf.alice, SPINNING)
        expected = p - replace(cx.act(op, tau, q), frame=SPINNING)
        np.testing.assert_allclose(
            _km(p.subtract(q, tau=tau)), _km(expected), atol=1e-12
        )
        # `add`: a displacement between two of Alice's points maps to the
        # difference of their images.
        q2 = _pt(a, frame=cxf.alice)
        moved = cx.act(op, tau, q2) - cx.act(op, tau, q)
        assert _close(p.add(q2 - q, tau=tau), p + replace(moved, frame=SPINNING))

    def test_time_dependent_frame_needs_tau(self) -> None:
        p = _pt((1, 2, 3), frame=SPINNING)
        q = _pt((1, 2, 3), frame=cxf.alice)
        with pytest.raises(TypeError, match="tau="):
            p.subtract(q)
        with pytest.raises(TypeError, match="tau="):
            p.add(q - q)

    def test_same_frame_ignores_tau(self) -> None:
        p, q = _pt((1, 2, 3)), _pt((3, 2, 1))
        assert p.subtract(q, tau=u.Q(1.0, "s")) == p - q
        assert p.add(q - p, tau=u.Q(1.0, "s")) == p + (q - p)


# A frame spinning about z relative to Alice's: its transition is time-dependent.
SPINNING = cxf.TransformedReferenceFrame(
    cxf.alice,
    cxfm.TimeDep.from_(lambda t: cxfm.Rotate.from_euler("Z", u.Q(10.0, "deg/s") * t)),
)


@given(a=_xyz(), b=_xyz(), c1=_charts, c2=_charts)
def test_difference_round_trip(a, b, c1, c2) -> None:
    """``(p - q) + q == p``, whatever charts the two points use."""
    p, q = _pt(a, c1), _pt(b, c2)
    assert _close((p - q) + q, p)
    assert _close(q + (p - q), p)


@given(a=_xyz(), b=_xyz(), c1=_charts, c2=_charts)
def test_difference_is_chart_invariant(a, b, c1, c2) -> None:
    """Curvilinear charts give the Cartesian difference."""
    expected = np.subtract(a, b)
    np.testing.assert_allclose(_km(_pt(a, c1) - _pt(b, c2)), expected, atol=1e-9)


@given(a=_xyz(), b=_xyz(), c=_charts)
def test_difference_is_antisymmetric(a, b, c) -> None:
    p, q = _pt(a, c), _pt(b, c)
    np.testing.assert_allclose(_km(p - q), -_km(q - p), atol=1e-9)
    np.testing.assert_allclose(_km(p - p), 0.0, atol=1e-9)


@given(a=_xyz(), b=_xyz(), r=_xyz(), c=_charts)
def test_differences_chain(a, b, r, c) -> None:
    """``(p - q) + (q - r) == p - r``."""
    p, q, s = _pt(a, c), _pt(b, c), _pt(r, c)
    np.testing.assert_allclose(_km((p - q) + (q - s)), _km(p - s), atol=1e-9)


@given(a=_xyz(), b=_xyz(), shift=_xyz(), c=_charts)
def test_difference_is_translation_invariant(a, b, shift, c) -> None:
    """A `Translate` moves both points and leaves their difference alone."""
    T = cxfm.Translate.from_(list(shift), "km")
    p, q = _pt(a, c), _pt(b, c)
    np.testing.assert_allclose(
        _km(cx.act(T, None, p) - cx.act(T, None, q)), _km(p - q), atol=1e-9
    )


@given(a=_xyz(), b=_xyz(), angle=st.floats(-180, 180), c=_charts)
def test_difference_is_rotation_equivariant(a, b, angle, c) -> None:
    """``R p - R q == R (p - q)``."""
    R = cxfm.Rotate.from_euler("Z", u.Q(angle, "deg"))
    p, q = _pt(a, c), _pt(b, c)
    lhs = cx.act(R, None, p) - cx.act(R, None, q)
    np.testing.assert_allclose(_km(lhs), _km(cx.act(R, None, p - q)), atol=1e-9)


@given(
    a=_xyz(),
    d=_xyz(),
    c=_charts,
    basis=st.sampled_from([cxr.coord_basis, cxr.phys_basis]),
)
def test_curvilinear_displacement_is_read_at_p(a, d, c, basis) -> None:
    """A displacement in ``c``, read at ``p``, moves ``p`` as its Cartesian form."""
    assume(not _phys_basis_broken(c, basis))
    p = _pt(a, c)
    dc = _dpl(*d).cconvert(c, at=p.cconvert(cxc.cart3d))
    dc = cxr.change_basis(dc, basis, at=p)
    assert _close(p + dc, p + _dpl(*d))


@given(a=_xyz(), c=_charts, basis=st.sampled_from([cxr.coord_basis, cxr.phys_basis]))
def test_position_vector_is_p_minus_origin(a, c, basis) -> None:
    """The position vector in any chart and basis is ``p - O`` in Cartesian."""
    assume(not _phys_basis_broken(c, basis))
    p = _pt(a, c)
    pv = cxr.change_basis(p.position_vector(basis), cxr.coord_basis, at=p)
    pv = pv.cconvert(cxc.cart3d, at=p)
    np.testing.assert_allclose(_km(pv), a, atol=1e-9)


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


@pytest.mark.parametrize("op", [operator.mul, operator.truediv])
def test_vector_operand_is_not_reported_as_scaling(op) -> None:
    """A vector operand is not a scalar: defer, so Python reports it plainly."""
    p = _cart_vec(1.0, 2.0, 3.0)
    c = cx.Coordinate(point=p, velocity=_cart_tangent(1, 2, 3))
    for lhs, rhs in [(p, p), (c, c), (c, p), (p, c)]:
        with pytest.raises(TypeError, match="unsupported operand"):
            op(lhs, rhs)


def test_point_scaling_error_fixes_work() -> None:
    """The fixes the error message suggests work in a curvilinear chart."""
    p = _sph_vec(1.0, 0.5, 0.5)
    q = p.cconvert(cxc.cart3d)
    # `k * p.position_vector()` scales the position about the origin.
    pos = (2 * p.position_vector()).cconvert(cxc.cart3d, at=p)
    # `k * (p - o)` scales a displacement.
    o = cx.Point.from_([0.0, 0.0, 0.0], "km")
    disp = 2 * (p - o)
    for out in (pos, disp):
        for n in ("x", "y", "z"):
            np.testing.assert_allclose(
                u.ustrip("km", out[n]), 2 * u.ustrip("km", q[n]), rtol=1e-12, atol=1e-12
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
