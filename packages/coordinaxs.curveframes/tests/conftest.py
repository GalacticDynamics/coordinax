"""Shared curves and fixtures for the `coordinaxs.curveframes` tests.

The Frenet-Serret and Bishop frames are two implementations of
`AbstractParallelTransportTransform`, so most of what the suite checks is the
*contract* they share rather than anything specific to either. The curves and
the per-type spec live here so that contract can be written once
(`test_parallel_transport_contract.py`) and each type's closed-form values
stay in its own module.
"""

__all__: tuple[str, ...] = ()

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import coordinax.frames as cxf
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


def as_arr(x: object, unit: str) -> np.ndarray:
    """Strip *x* to a plain float array in *unit*."""
    assert isinstance(x, u.AbstractQuantity)
    return np.asarray(u.ustrip(unit, x), dtype=float)


@pytest.fixture
def arr():
    """The `as_arr` helper, as a fixture."""
    return as_arr


# ===================================================================
# Per-type specs
#
# `atol` differs by construction, not by accident: Frenet-Serret is closed-form
# autodiff, while Bishop integrates a parallel-transport ODE and so carries
# integrator error. Each spec states the tolerance its own frame can hold.

#: Per-assertion absolute tolerances, per type.
#
# These are not stylistic. Each entry is the value that assertion carried in the
# per-type modules before they were merged, and the two columns differ by up to
# four orders of magnitude: Frenet-Serret's `act` plumbing agrees to 1e-10
# because it is closed-form autodiff, while Bishop's is 1e-5 because a transport
# ODE has been integrated in between. A single shared tolerance would have to be
# the looser of the two everywhere, which would stop Frenet-Serret regressions
# from being caught at all.
TOLERANCES = {
    "frenet-serret": {
        "location": 1e-6,  # gamma(0), straight from the curve
        "field": 1e-5,  # a frame-field value or norm
        "orthogonality": 1e-5,  # dot products within the triad
        "transform_roundtrip": 1e-4,  # R_inv @ (R @ (p - g) - g_inv)
        "double_inverse": 1e-4,
        "act": 1e-6,  # one `act` application
        "act_roundtrip": 1e-6,  # act then act-inverse
        "chain": 1e-6,  # through one intermediate frame
        "full_chain": 1e-5,  # Alice -> frame -> Alex -> frame -> Alice
        "plumbing": 1e-10,  # jit vs eager; transition vs direct xop
    },
    "bishop": {
        "location": 1e-6,
        "field": 1e-5,
        "orthogonality": 1e-4,
        "transform_roundtrip": 1e-3,
        "double_inverse": 1e-4,
        "act": 1e-5,
        "act_roundtrip": 1e-3,
        "chain": 1e-3,
        "full_chain": 1e-2,
        "plumbing": 1e-5,
    },
}

PARALLEL_TRANSPORT_TYPES = {
    "frenet-serret": SimpleNamespace(
        transform_cls=cxfc.FrenetSerretTransform,
        frame_cls=cxfc.FrenetSerretFrame,
        triad=("tangent", "normal", "binormal"),
    ),
    "bishop": SimpleNamespace(
        transform_cls=cxfc.BishopTransform,
        frame_cls=cxfc.BishopFrame,
        triad=("tangent", "normal1", "normal2"),
    ),
}


@pytest.fixture(params=sorted(PARALLEL_TRANSPORT_TYPES))
def pt_case(request: pytest.FixtureRequest) -> SimpleNamespace:
    """Every parallel-transport frame type, on the unit circle.

    Yields a namespace of ``transform``, ``frame``, ``tol`` (the per-assertion
    tolerance table above) and ``fields`` -- a callable
    ``(transform, tau) -> (e0, e1, e2)`` returning that type's three frame
    fields in right-handed order, so contract tests can be written without
    naming ``binormal`` or ``normal2``.
    """
    spec = PARALLEL_TRANSPORT_TYPES[request.param]

    def fields(transform: object, tau: u.AbstractQuantity) -> tuple:
        return tuple(getattr(transform, name)(tau) for name in spec.triad)

    return SimpleNamespace(
        name=request.param,
        transform=spec.transform_cls.from_curve(circle),
        frame=spec.frame_cls.from_curve(cxf.Alice(), circle),
        triad=spec.triad,
        tol=SimpleNamespace(**TOLERANCES[request.param]),
        fields=fields,
    )


@pytest.fixture
def curve():
    """The unit-circle curve function itself (for constructor tests)."""
    return circle


# ===================================================================
# Single-type fixtures (closed-form value tests)


@pytest.fixture
def circle_fs() -> cxfc.FrenetSerretTransform:
    """Frenet-Serret transform on the unit circle."""
    return cxfc.FrenetSerretTransform.from_curve(circle)


@pytest.fixture
def circle_yr_fs() -> cxfc.FrenetSerretTransform:
    """Frenet-Serret transform on the opaque-unit (yr) circle."""
    return cxfc.FrenetSerretTransform.from_curve(circle_yr, tau_unit="yr")


@pytest.fixture
def circle_bishop() -> cxfc.BishopTransform:
    """Bishop transform on the unit circle."""
    return cxfc.BishopTransform.from_curve(circle)


@pytest.fixture
def line_bishop() -> cxfc.BishopTransform:
    """Bishop transform on a straight line (kappa=0)."""
    return cxfc.BishopTransform.from_curve(straight_line)


@pytest.fixture
def helix_bishop() -> cxfc.BishopTransform:
    """Bishop transform on a helix."""
    return cxfc.BishopTransform.from_curve(helix)


@pytest.fixture
def circle_yr_bishop() -> cxfc.BishopTransform:
    """Bishop transform on the opaque-unit (yr) circle."""
    return cxfc.BishopTransform.from_curve(circle_yr, tau_unit="yr")


@pytest.fixture
def circle_fs_frame() -> cxfc.FrenetSerretFrame:
    """Frenet-Serret frame on the unit circle, relative to Alice."""
    return cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), circle)


@pytest.fixture
def circle_bishop_frame() -> cxfc.BishopFrame:
    """Bishop frame on the unit circle, relative to Alice."""
    return cxfc.BishopFrame.from_curve(cxf.Alice(), circle)


@pytest.fixture
def line_bishop_frame() -> cxfc.BishopFrame:
    """Bishop frame on a straight line, relative to Alice."""
    return cxfc.BishopFrame.from_curve(cxf.Alice(), straight_line)
