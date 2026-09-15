"""`SignedPlanarBuilder` keeps an f32 curve in f32, `_float`-style.

`_plane_normal`'s docstring claims taking `dtype` from the caller (rather than
defaulting it) keeps an f32 curve in f32, and `_planar_tol`'s only doctest
exercises the f64 branch. This pins the f32 half of both claims: an f32 curve
yields an f32 rotation matrix and `signed_curvature`, and `_planar_tol` gives
the f32 value its docstring states, not just the f64 one.
"""

import jax.numpy as jnp
import numpy as np

import unxt as u

from coordinaxs.curveframes._src.signedplanar import SignedPlanarBuilder, _planar_tol


def circle_f32(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Unit circle in the xy-plane, forced to f32 regardless of x64."""
    t = tau.ustrip("s").astype(jnp.float32)
    return u.Q(
        jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]).astype(jnp.float32), "km"
    )


def test_f32_curve_yields_f32_rotation_matrix() -> None:
    b = SignedPlanarBuilder(circle_f32, "s")
    R = b.rotation_matrix(u.Q(jnp.float32(0.3), "s"))
    assert R.dtype == jnp.float32


def test_f32_curve_yields_f32_signed_curvature() -> None:
    b = SignedPlanarBuilder(circle_f32, "s")
    k = b.signed_curvature(u.Q(jnp.float32(0.3), "s"))
    assert k.value.dtype == jnp.float32


def test_planar_tol_f32() -> None:
    np.testing.assert_allclose(float(_planar_tol(jnp.float32(1.0))), 3.45e-4, rtol=1e-2)


def test_planar_tol_f64() -> None:
    np.testing.assert_allclose(float(_planar_tol(jnp.float64(1.0))), 1.49e-8, rtol=1e-2)
