"""Shared curves and fixtures for the `coordinaxs.curveframes` tests.

The Frenet-Serret and Bishop frames are two implementations of
`AbstractCurveFrameBuilder`, wrapped by `coordinax.transforms.TimeDep`, so
most of what the suite checks is the *contract* they share rather than anything
specific to either. The curves and the per-type spec live here so that contract
can be written once (``test_parallel_transport_contract.py``) and each type's
closed-form values stay in its own module.
"""

__all__: tuple[str, ...] = ()

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

import coordinax.frames as cxf
import coordinax.transforms as cxfm
import quaxed.numpy as qnp
import unxt as u

import coordinaxs.curveframes as cxfc

# ===================================================================
# Curves


def circle(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Unit circle in the xy-plane, radius 1 km, period 2*pi s."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


def straight_line(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Straight line along the x-axis (kappa=0 everywhere).

    The Frenet-Serret frame is singular on this curve; Bishop is not.
    """
    t = tau.ustrip("s")
    return u.Q(jnp.stack([t, jnp.zeros_like(t), jnp.zeros_like(t)]), "km")


def helix(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Helix with pitch along the z-axis."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


def circle_yr(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Circle of radius 5 km with angular speed 2*pi rad/yr.

    The curve converts tau to radians internally, so its "natural" tau-unit is
    opaque to the caller -- a harder case than `circle`.
    """
    omega = u.Q(2 * jnp.pi, "rad/yr")
    phase = (omega * tau).uconvert("rad").ustrip("rad")
    x = u.Q(5, "km") * jnp.cos(phase)
    y = u.Q(5, "km") * jnp.sin(phase)
    z = u.Q(0, "km") * jnp.ones_like(phase)
    return qnp.stack([x, y, z], axis=-1)


# ===================================================================
# Helpers


def inverse_rotation(builder: object, tau: u.AbstractQuantity) -> object:
    """Rotation matrix of the inverse family at ``tau`` (i.e. R^T).

    The rows of this matrix are the inverse frame's triad, i.e. the *columns*
    of the forward R.
    """
    # The forward family is ``Translate(-gamma) | Rotate(R)``, so its inverse
    # reverses the pipe and the rotation lands first. Pin that ordering: a
    # silent reversal would otherwise surface as an opaque AttributeError.
    inv = cxfm.TimeDep(builder).inverse.evaluate_at(tau)[0]
    assert isinstance(inv, cxfm.Rotate)
    return inv.R


# ===================================================================
# Per-type specs

#: Absolute tolerances, in four tiers per type.
#
# The columns differ by construction, not by accident, and must not be merged:
# Frenet-Serret is closed-form autodiff, while Bishop integrates a
# parallel-transport ODE and carries integrator error, so a single shared
# column would have to be Bishop's everywhere and would stop Frenet-Serret
# regressions from being caught at all.
#
# `tight`: gamma itself (the curve, evaluated directly).
# `plumbing`: jit vs eager, vmap vs eager, transition vs direct xop. Kept
#   separate from `tight` because Bishop's jit/vmap path re-runs the
#   parallel-transport ODE solve, which is the likeliest source of
#   platform-dependent noise in this suite -- it must not be pulled down to
#   `tight`'s value just because the two happen to coincide for Frenet-Serret.
# `field`: a frame-field value, norm, dot product, or a single `act`.
# `loose`: a multi-step chain, where the per-step error compounds.
TOLERANCES = {
    "frenet-serret": {"tight": 1e-10, "plumbing": 1e-10, "field": 1e-6, "loose": 1e-5},
    "bishop": {"tight": 1e-6, "plumbing": 1e-5, "field": 1e-5, "loose": 1e-3},
}

PARALLEL_TRANSPORT_TYPES = {
    "frenet-serret": SimpleNamespace(
        builder_cls=cxfc.FrenetSerretBuilder,
        frame_cls=cxfc.FrenetSerretFrame,
        triad=("tangent", "normal", "binormal"),
    ),
    "bishop": SimpleNamespace(
        builder_cls=cxfc.BishopBuilder,
        frame_cls=cxfc.BishopFrame,
        triad=("tangent", "normal1", "normal2"),
    ),
}


@pytest.fixture(params=sorted(PARALLEL_TRANSPORT_TYPES))
def pt_case(request: pytest.FixtureRequest) -> SimpleNamespace:
    """Every parallel-transport frame type, on the unit circle.

    Yields a namespace of ``builder`` (the `AbstractCurveFrameBuilder`),
    ``xop`` (the `coordinax.transforms.TimeDep` wrapping it), ``frame``,
    ``tol`` (the tolerance tiers above), ``yr_builder`` (the same type on the
    opaque-unit `circle_yr`) and ``fields`` -- a callable
    ``(builder, tau) -> (e0, e1, e2)`` returning that type's three frame
    fields in right-handed order, so contract tests can be written without
    naming ``binormal`` or ``normal2``.
    """
    spec = PARALLEL_TRANSPORT_TYPES[request.param]
    builder = spec.builder_cls(circle)

    def fields(bldr: object, tau: u.AbstractQuantity) -> tuple:
        return tuple(getattr(bldr, name)(tau) for name in spec.triad)

    return SimpleNamespace(
        name=request.param,
        builder=builder,
        builder_cls=spec.builder_cls,
        yr_builder=spec.builder_cls(circle_yr, "yr"),
        xop=cxfm.TimeDep(builder),
        frame=spec.frame_cls.from_curve(cxf.Alice(), circle),
        frame_cls=spec.frame_cls,
        triad=spec.triad,
        tol=SimpleNamespace(**TOLERANCES[request.param]),
        fields=fields,
    )
