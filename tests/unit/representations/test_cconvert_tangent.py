"""Tests for cconvert dispatching to tangent_map when source is TangentGeometry."""

__all__: tuple[str, ...] = ()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr

usys = u.unitsystems.si


class TestCconvertTangentGeometry:
    """cconvert with TangentGeometry representation dispatches to tangent_map."""

    def test_same_chart_noncartesian_matches_change_basis(self) -> None:
        """Same-chart tangent conversion should reduce to basis conversion."""
        v = {"r": jnp.array(5), "theta": jnp.array(1), "phi": jnp.array(2)}
        at = {"r": jnp.array(3), "theta": jnp.array(0.5), "phi": jnp.array(0)}

        result = cxr.cconvert(
            v, cxc.sph3d, cxr.coord_disp, cxc.sph3d, cxr.phys_disp, at=at, usys=usys
        )
        expected = cxr.change_basis(
            v, cxc.sph3d, cxr.coord_basis, cxr.phys_basis, at=at, usys=usys
        )

        np.testing.assert_allclose(result["r"], expected["r"])
        np.testing.assert_allclose(result["theta"], expected["theta"])
        np.testing.assert_allclose(result["phi"], expected["phi"])

    def test_cart2d_to_polar2d_coord_disp(self) -> None:
        """Cconvert with coord_disp routes through tangent_map (Jacobian)."""
        v = {"x": jnp.array(1), "y": jnp.array(0)}
        at = {"x": jnp.array(1), "y": jnp.array(0)}
        result = cxr.cconvert(
            v, cxc.cart2d, cxr.coord_disp, cxc.polar2d, cxr.coord_disp, at=at, usys=usys
        )
        np.testing.assert_allclose(result["r"], 1, atol=1e-6)
        np.testing.assert_allclose(result["theta"], 0, atol=1e-6)

    def test_same_chart_identity(self) -> None:
        """Cconvert with same chart + TangentGeometry returns input unchanged."""
        v = {"x": jnp.array(2), "y": jnp.array(3)}
        at = {"x": jnp.array(1), "y": jnp.array(0)}
        result = cxr.cconvert(
            v, cxc.cart2d, cxr.coord_disp, cxc.cart2d, cxr.coord_disp, at=at, usys=usys
        )
        np.testing.assert_allclose(result["x"], 2)
        np.testing.assert_allclose(result["y"], 3)

    def test_same_chart_cartesian_without_at(self) -> None:
        """Cartesian same-chart basis conversion should not require `at`."""
        v = {"x": jnp.array(1), "y": jnp.array(2)}
        result = cxr.cconvert(
            v, cxc.cart2d, cxr.coord_disp, cxc.cart2d, cxr.phys_disp, usys=usys
        )
        np.testing.assert_allclose(result["x"], v["x"])
        np.testing.assert_allclose(result["y"], v["y"])

    def test_cart3d_to_sph3d_coord_vel(self) -> None:
        """Cconvert with coord_vel representation uses tangent_map semantics."""
        v = {"x": jnp.array(1), "y": jnp.array(0), "z": jnp.array(0)}
        at = {"x": jnp.array(1), "y": jnp.array(0), "z": jnp.array(0)}
        result = cxr.cconvert(
            v, cxc.cart3d, cxr.coord_vel, cxc.sph3d, cxr.coord_vel, at=at, usys=usys
        )
        # Purely radial result
        np.testing.assert_allclose(result["r"], 1, atol=1e-6)
        np.testing.assert_allclose(result["theta"], 0, atol=1e-6)
        np.testing.assert_allclose(result["phi"], 0, atol=1e-6)

    def test_same_chart_respects_tangent_semantic_kind(self) -> None:
        """Displacement and velocity variants should follow the same basis map."""
        v = {"r": jnp.array(5), "theta": jnp.array(1), "phi": jnp.array(2)}
        at = {"r": jnp.array(3), "theta": jnp.array(0.5), "phi": jnp.array(0)}

        out_disp = cxr.cconvert(
            v, cxc.sph3d, cxr.coord_disp, cxc.sph3d, cxr.phys_disp, at=at, usys=usys
        )
        out_vel = cxr.cconvert(
            v, cxc.sph3d, cxr.coord_vel, cxc.sph3d, cxr.phys_vel, at=at, usys=usys
        )

        np.testing.assert_allclose(out_disp["r"], out_vel["r"])
        np.testing.assert_allclose(out_disp["theta"], out_vel["theta"])
        np.testing.assert_allclose(out_disp["phi"], out_vel["phi"])

    def test_jit_compatible(self) -> None:
        """Cconvert with TangentGeometry is JIT-compatible."""
        v = {"x": jnp.array(1), "y": jnp.array(0)}
        at = {"x": jnp.array(1), "y": jnp.array(0)}

        @jax.jit
        def run(v, at):
            return cxr.cconvert(
                v,
                cxc.cart2d,
                cxr.coord_disp,
                cxc.polar2d,
                cxr.coord_disp,
                at=at,
                usys=usys,
            )

        result = run(v, at)
        np.testing.assert_allclose(result["r"], 1, atol=1e-6)

    def test_round_trip(self) -> None:
        """Cconvert tangent round trip: cart2d → polar2d → cart2d is identity."""
        v_cart = {"x": jnp.array(1), "y": jnp.array(0)}
        at_cart = {"x": jnp.array(1), "y": jnp.array(0)}

        # cart → polar
        v_polar = cxr.cconvert(
            v_cart,
            cxc.cart2d,
            cxr.coord_disp,
            cxc.polar2d,
            cxr.coord_disp,
            at=at_cart,
            usys=usys,
        )

        # at in polar coords
        at_polar = cxr.cconvert(at_cart, cxc.cart2d, cxr.point, cxc.polar2d, usys=usys)

        # polar → cart
        v_cart_back = cxr.cconvert(
            v_polar,
            cxc.polar2d,
            cxr.coord_disp,
            cxc.cart2d,
            cxr.coord_disp,
            at=at_polar,
            usys=usys,
        )

        np.testing.assert_allclose(v_cart_back["x"], v_cart["x"], atol=1e-6)
        np.testing.assert_allclose(v_cart_back["y"], v_cart["y"], atol=1e-6)


class TestCconvertAtRequired:
    """cconvert with TangentGeometry requires the `at` keyword argument."""

    def test_at_required_for_nonlinear_charts(self) -> None:
        """Missing `at` raises informative error for non-Cartesian charts."""
        v = {"x": jnp.array(1), "y": jnp.array(0)}
        with pytest.raises((TypeError, ValueError)):
            cxr.cconvert(
                v, cxc.cart2d, cxr.coord_disp, cxc.polar2d, cxr.coord_disp, usys=usys
            )


_T = 0.7


def _gamma(t):
    """A helix in cart3d: the trajectory both transformation laws are read off."""
    return jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t])


def _to_sph(xyz):
    x, y, z = xyz
    r = jnp.sqrt(x**2 + y**2 + z**2)
    return jnp.stack([r, jnp.arccos(z / r), jnp.arctan2(y, x)])


def _sph_of_t(t):
    return _to_sph(_gamma(t))


_XYZ = _gamma(_T)
_V = jax.jacfwd(_gamma)(_T)
_A = jax.jacfwd(jax.jacfwd(_gamma))(_T)
_AT = dict(zip("xyz", _XYZ, strict=True))


def _convert(vec, kind):
    """Convert a cart3d tangent vector to sph3d, at the point on the curve."""
    v = dict(zip("xyz", vec, strict=True))
    got = cxr.cconvert(v, cxc.cart3d, kind, cxc.sph3d, kind, at=_AT, usys=usys)
    return np.asarray([float(got[k]) for k in ("r", "theta", "phi")])


class TestAccelerationPushesForwardAsAVector:
    """An acceleration is a tangent vector, so it converts as ``J a``.

    The two differ by the Christoffel term, so they disagree in spherical
    coordinates on flat R^3.
    """

    def test_it_is_the_linear_pushforward(self):
        """``J a`` exactly -- the transformation law for a tangent vector."""
        want = np.asarray(jax.jacfwd(_to_sph)(_XYZ) @ _A)
        got = _convert(_A, cxr.coord_acc)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-14)

    def test_it_is_not_the_coordinate_second_derivative(self):
        """The two differ by the Christoffel term, here on *flat* R^3.

        If this ever starts passing, `cconvert` has begun adding the
        non-tensorial term and accelerations no longer transform as vectors.
        """
        qddot = np.asarray(jax.jacfwd(jax.jacfwd(_sph_of_t))(_T))
        assert np.max(np.abs(_convert(_A, cxr.coord_acc) - qddot)) > 0.5

    def test_velocity_has_no_such_gap(self):
        """First derivatives *are* tangent vectors, so velocity agrees with d/dt."""
        want = np.asarray(jax.jacfwd(_sph_of_t)(_T))
        got = _convert(_V, cxr.coord_vel)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-14)
