"""`_param` promotes the curve parameter to float, *preserving* f32.

`Quantity.astype(float)` names the **default** float, so under
`JAX_ENABLE_X64=1` it silently widens an f32 `tau` or `station` to f64 and
discards a deliberate choice of single precision. Four accessors used to cast
that way independently; the promotion now lives in `_param`, the one funnel
every accessor takes its parameter from, so all three builders agree.

See GalacticDynamics/coordinax#886.
"""

from typing import Any

import jax.numpy as jnp
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc


def circle(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """A circle that does NOT force a dtype -- it inherits from ``tau``.

    Forcing f32 inside the curve would mask a parameter widened on the way
    in, because the curve casts it straight back. That is exactly why the
    pre-existing dtype tests did not catch #886.
    """
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


BUILDERS = [cxfc.FrenetSerretBuilder, cxfc.BishopBuilder, cxfc.SignedPlanarBuilder]
IDS = ["frenet-serret", "bishop", "signed-planar"]

#: `BishopBuilder.rotation_matrix` is excluded from the f32 assertions below,
#: and that is by design rather than an unfixed gap. Its triad comes out of the
#: parallel-transport ODE, seeded at ``tau_0`` and integrated by `diffrax` at a
#: default ``rtol = atol = 1e-10`` -- three orders of magnitude below f32's
#: ``eps`` of ~1.2e-7, so the solve could not meet its own tolerance in single
#: precision. The *parameter* still round-trips as f32 (`tangent`, `location`
#: below); it is the transport that is f64.
CLOSED_FORM = [cxfc.FrenetSerretBuilder, cxfc.SignedPlanarBuilder]
CLOSED_FORM_IDS = ["frenet-serret", "signed-planar"]


@pytest.mark.parametrize("builder_cls", BUILDERS, ids=IDS)
class TestF32ParameterIsPreserved:
    """An f32 parameter must stay f32 through every accessor."""

    def test_tangent(self, builder_cls: Any) -> None:
        """Inherited from `base.tangent`, which cast independently."""
        T = builder_cls(circle, "s").tangent(u.Q(jnp.float32(0.3), "s"))
        assert T.value.dtype == jnp.float32

    def test_location(self, builder_cls: Any) -> None:
        loc = builder_cls(circle, "s").location(u.Q(jnp.float32(0.3), "s"))
        assert loc.value.dtype == jnp.float32

    def test_f32_station(self, builder_cls: Any) -> None:
        """A pinned f32 `station` is the other way the parameter arrives."""
        b = builder_cls(circle, "s", u.Q(jnp.float32(0.4), "s"))
        assert b.tangent(u.Q(jnp.float32(0.0), "s")).value.dtype == jnp.float32


@pytest.mark.parametrize("builder_cls", CLOSED_FORM, ids=CLOSED_FORM_IDS)
class TestF32RotationMatrixClosedForm:
    """The closed-form builders keep the whole triad in f32."""

    def test_from_tau(self, builder_cls: Any) -> None:
        R = builder_cls(circle, "s").rotation_matrix(u.Q(jnp.float32(0.3), "s"))
        assert R.dtype == jnp.float32

    def test_from_station(self, builder_cls: Any) -> None:
        b = builder_cls(circle, "s", u.Q(jnp.float32(0.4), "s"))
        assert b.rotation_matrix(u.Q(jnp.float32(0.0), "s")).dtype == jnp.float32


def test_bishop_transport_is_f64_by_design() -> None:
    """Pinned deliberately: the ODE cannot meet ``rtol=1e-10`` in f32.

    If a future change makes this f32, the solve is silently no longer
    meeting its stated tolerance -- so this asserting f64 is the guard, not
    an admission.
    """
    R = cxfc.BishopBuilder(circle, "s").rotation_matrix(u.Q(jnp.float32(0.3), "s"))
    assert R.dtype == jnp.float64


@pytest.mark.parametrize("builder_cls", BUILDERS, ids=IDS)
class TestIntegerParameterStillPromotes:
    """Preserving f32 must not stop ints from becoming floats."""

    def test_rotation_matrix(self, builder_cls: Any) -> None:
        R = builder_cls(circle, "s").rotation_matrix(u.Q(jnp.asarray(1), "s"))
        assert jnp.issubdtype(R.dtype, jnp.floating)

    def test_tangent(self, builder_cls: Any) -> None:
        T = builder_cls(circle, "s").tangent(u.Q(jnp.asarray(1), "s"))
        assert jnp.issubdtype(T.value.dtype, jnp.floating)

    def test_location(self, builder_cls: Any) -> None:
        """`location` did not promote before; under #886's fix it does."""
        loc = builder_cls(circle, "s").location(u.Q(jnp.asarray(1), "s"))
        assert jnp.issubdtype(loc.value.dtype, jnp.floating)


def test_f64_is_untouched() -> None:
    """The default path must not change: f64 in, f64 out."""
    R = cxfc.FrenetSerretBuilder(circle, "s").rotation_matrix(u.Q(0.3, "s"))
    assert R.dtype == jnp.float64
