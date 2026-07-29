"""Closed-form Frenet-Serret values on the unit circle.

Structural guarantees shared with Bishop -- orthonormality, right-handedness,
inverse semantics, `act`, `frame_transition`, jit/vmap -- are asserted once in
`test_parallel_transport_contract.py`. What is left here is what is specific to
Frenet-Serret: the actual (T, N, B) values, which Bishop does not share because
its normals are parallel-transported rather than curvature-derived.

For a unit circle at tau=0:
    gamma = (1, 0, 0), T = (0, 1, 0), N = (-1, 0, 0), B = (0, 0, 1)
    R = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    inv_location = -R @ gamma = -[T.g, N.g, B.g] = (0, 1, 0)
    inv_T = col 0 of R = (0, -1, 0)
    inv_N = col 1 of R = (1, 0, 0)
    inv_B = col 2 of R = (0, 0, 1)
"""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import numpy as np
import pytest

import coordinax.transforms as cxfm
import unxt as u

import coordinaxs.curveframes as cxfc


class TestFrenetSerretTriadValues:
    """T, N, B take their curvature-derived values on the unit circle."""

    def test_tangent_is_dimensionless(self, circle_fs: cxfc.FrenetSerretTransform):
        """The raw derivative carries km/s; after normalisation it is unitless."""
        assert circle_fs.tangent(u.Q(0, "s")).unit == u.unit("")

    @pytest.mark.parametrize(
        ("field", "tau_val", "expected"),
        [
            ("normal", 0, [-1, 0, 0]),  # points inward
            ("normal", jnp.pi / 2, [0, -1, 0]),
            ("binormal", 0, [0, 0, 1]),  # circle lies in the xy-plane
            ("binormal", 2, [0, 0, 1]),  # ... at every tau
        ],
    )
    def test_field_value(
        self,
        circle_fs: cxfc.FrenetSerretTransform,
        field: str,
        tau_val: float,
        expected: list[float],
    ):
        got = getattr(circle_fs, field)(u.Q(tau_val, "s"))
        np.testing.assert_allclose(got.value, expected, atol=1e-5)


class TestFrenetSerretInverseValues:
    """The inverse frame fields are the columns of R (see module docstring)."""

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("location", [0, 1, 0]),
            ("tangent", [0, -1, 0]),
            ("normal", [1, 0, 0]),
            ("binormal", [0, 0, 1]),
        ],
    )
    def test_inverse_field_at_zero(
        self, circle_fs: cxfc.FrenetSerretTransform, field: str, expected: list[float]
    ):
        got = getattr(circle_fs.inverse, field)(u.Q(0, "s"))
        np.testing.assert_allclose(got.value, expected, atol=1e-5)


class TestFrenetSerretOpaqueUnits:
    """A curve whose internal unit (yr) differs from the caller's."""

    @pytest.mark.parametrize(
        ("field", "expected"),
        [("tangent", [0, 1, 0]), ("normal", [-1, 0, 0]), ("binormal", [0, 0, 1])],
    )
    def test_field_at_zero(
        self,
        circle_yr_fs: cxfc.FrenetSerretTransform,
        field: str,
        expected: list[float],
    ):
        got = getattr(circle_yr_fs, field)(u.Q(0, "yr"))
        np.testing.assert_allclose(got.value, expected, atol=1e-5)

    def test_inverse_location_at_zero(self, circle_yr_fs: cxfc.FrenetSerretTransform):
        """gamma=(5,0,0) km, T=(0,1,0), N=(-1,0,0) => inv_location = (0, 5, 0)."""
        loc = circle_yr_fs.inverse.location(u.Q(0, "yr"))
        np.testing.assert_allclose(loc.value, [0, 5, 0], atol=1e-3)


class TestFrenetSerretAct:
    """`act` values that depend on the Frenet-Serret triad specifically."""

    def test_act_forward_off_curve(self, circle_fs: cxfc.FrenetSerretTransform, arr):
        """At tau=0, p=(2,0,0) km => delta=(1,0,0) => R @ delta = (0,-1,0) km."""
        result = cxfm.act(circle_fs, u.Q(0, "s"), u.Q(jnp.array([2, 0, 0]), "km"))
        np.testing.assert_allclose(arr(result, "km"), [0, -1, 0], atol=1e-6)
