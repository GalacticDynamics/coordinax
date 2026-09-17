r"""`SweptTube` owns the slice family, and with it the n-plane gauge.

The gauge is *director* data -- a Cosserat frame -- and a curve does not carry
it. Criterion 11 below is the whole argument: one curve with no time dependence
at all, two directors, two different correct answers. No rule derived from the
curve can produce both, which is why the library refuses to guess (#829, #870).
"""

from jaxtyping import TypeCheckError
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc

BOUNDS = (u.Q(-1.0, "km"), u.Q(2.0, "km"))
OFF_AXIS = {"tau": u.Q(0.5, "km"), "n1": u.Q(0.2, "km"), "n2": u.Q(0.1, "km")}

#: `a = omega . T(tau_0)` for the spun-director criterion.
SPIN = 1.0 / np.sqrt(1.16)

#: `T(tau_0)` for the helix below at `tau_0 = 0`, and an orthonormal pair
#: spanning its normal plane -- the frame a director is expressed in.
_T0 = np.asarray([0.0, 1.0, 0.4])
_T0 = _T0 / np.linalg.norm(_T0)
_E = np.asarray([1.0, 0.0, 0.0]) - np.dot(np.asarray([1.0, 0.0, 0.0]), _T0) * _T0
_E = _E / np.linalg.norm(_E)
_F = np.cross(_T0, _E)


def static_helix(tau: u.AbstractQuantity, t: u.AbstractQuantity) -> u.AbstractQuantity:
    """No ``t`` dependence whatsoever: the curve is identical at every time."""
    s = tau.ustrip("km")
    return u.Q(jnp.stack([jnp.cos(s), jnp.sin(s), 0.4 * s]), "km")


def _rotation(theta: Any) -> Any:
    """Rodrigues about a deliberately generic axis."""
    k = jnp.asarray(np.asarray([0.3, -0.5, 0.81]) / np.linalg.norm([0.3, -0.5, 0.81]))
    cross = jnp.asarray(
        [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=float
    )
    return jnp.eye(3) + jnp.sin(theta) * cross + (1 - jnp.cos(theta)) * (cross @ cross)


def rotating_helix(
    tau: u.AbstractQuantity, t: u.AbstractQuantity
) -> u.AbstractQuantity:
    """The same helix, carried bodily around a generic axis."""
    s = tau.ustrip("km")
    base = jnp.stack([jnp.cos(s), jnp.sin(s), 0.4 * s])
    return u.Q(_rotation(t.ustrip("s")) @ base, "km")


def _k(tube: cxfc.SweptTube, t: float = 0.3, point: dict | None = None) -> np.ndarray:
    at = OFF_AXIS if point is None else point
    return np.asarray(cxfc.rate_of_strain(tube, at, u.Q(t, "s")).value)


# --------------------------------------------------------------------------
# Criterion 11: the one the design exists for.


def test_a_static_curve_with_a_spun_director_strains() -> None:
    """One curve, two directors, two answers -- so no curve-derived rule works.

    `static_helix` has no `t` dependence at all, so every candidate seed rule
    ever proposed returns the same thing for both tubes below. A rod spinning
    about its own axis and one at rest trace the same curve and do not have the
    same rate of strain.
    """
    fixed = cxfc.SweptTube(
        static_helix, "km", tau_bounds=BOUNDS, director=lambda t: jnp.asarray(_E)
    )
    spun = cxfc.SweptTube(
        static_helix,
        "km",
        tau_bounds=BOUNDS,
        director=lambda t: jnp.cos(SPIN * t.ustrip("s")) * jnp.asarray(_E)
        + jnp.sin(SPIN * t.ustrip("s")) * jnp.asarray(_F),
    )

    assert _k(fixed, t=0.0)[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert _k(spun, t=0.0)[0, 0] == pytest.approx(-0.067526, abs=1e-5)


# --------------------------------------------------------------------------
# Criterion 1: the headline.


def test_a_rigid_rotation_with_a_carried_director_does_not_strain() -> None:
    """A rigid motion is an isometry. Generic axis, so nothing passes by luck."""
    carried = cxfc.SweptTube(
        rotating_helix,
        "km",
        tau_bounds=BOUNDS,
        director=lambda t: _rotation(t.ustrip("s")) @ jnp.asarray(_E),
    )

    assert np.abs(_k(carried)).max() < 1e-9


def test_a_director_left_behind_reports_the_drift_as_strain() -> None:
    """The defect #870 filed, now reachable only by choosing it.

    Same body, same isometry; a director that does not follow it. That this
    still returns a number is the point -- `SweptTube` makes the gauge the
    caller's explicit choice, it does not make every choice correct.
    """
    left_behind = cxfc.SweptTube(
        rotating_helix, "km", tau_bounds=BOUNDS, director=lambda t: jnp.asarray(_E)
    )

    assert np.abs(_k(left_behind)).max() > 1e-3


# --------------------------------------------------------------------------
# Criterion 5: the contract.


def test_the_bishop_path_requires_a_director() -> None:
    """Omission raises rather than silently defaulting. That is the whole buy."""
    with pytest.raises(ValueError, match="`director` is required"):
        cxfc.SweptTube(static_helix, "km", tau_bounds=BOUNDS)


def test_a_seedless_builder_refuses_a_director() -> None:
    """Frenet fixes N and B from the curve, so a seed would be ignored.

    Rejected here rather than one layer down: `FrenetSerretBuilder` does raise
    on `initial_normal`, but with a different error and only once a slice is
    built -- too late to name the argument that was wrong.
    """
    with pytest.raises(ValueError, match="takes no seed"):
        cxfc.SweptTube(
            static_helix,
            "km",
            tau_bounds=BOUNDS,
            director=lambda t: jnp.asarray(_E),
            builder=cxfc.FrenetSerretBuilder,
        )


def test_a_seedless_builder_needs_no_director() -> None:
    """The other half of the same rule."""
    tube = cxfc.SweptTube(
        static_helix, "km", tau_bounds=BOUNDS, builder=cxfc.FrenetSerretBuilder
    )

    assert tube(u.Q(0.0, "s")).components == ("tau", "n1", "n2")


def test_a_builder_instance_is_refused() -> None:
    """`builder=BishopBuilder(...)` is the natural slip: everything else is a value.

    Two layers catch it, and which one fires depends on the environment. With
    runtime typechecking on -- as under this repo's pytest config -- the
    ``builder: type`` annotation rejects the instance before `__check_init__`
    runs, as a `TypeCheckError`. With it off, `__check_init__` does, naming the
    argument and the fix. Unguarded it was neither: `issubclass() arg 1 must be
    a class`, which names nothing.
    """
    instance = cxfc.BishopBuilder(
        cxfc.AtTime(static_helix, u.Q(0.0, "s")), "km", initial_normal=jnp.asarray(_E)
    )
    with pytest.raises((TypeError, TypeCheckError)):
        cxfc.SweptTube(
            static_helix,
            "km",
            tau_bounds=BOUNDS,
            director=lambda t: jnp.asarray(_E),
            builder=instance,
        )


def test_an_unrelated_class_is_refused_rather_than_misdiagnosed() -> None:
    """The worse case: a *class* passes `issubclass` and gets a wrong answer.

    `builder=dict` reached the seedless branch and was told it "takes no seed",
    as though `dict` were a legitimate seedless builder.
    """
    with pytest.raises(TypeError, match="must be a builder"):
        cxfc.SweptTube(
            static_helix,
            "km",
            tau_bounds=BOUNDS,
            director=lambda t: jnp.asarray(_E),
            builder=dict,
        )


# --------------------------------------------------------------------------
# The narrowing, and criterion 6.


def test_a_swept_tube_needs_no_gauge_assertion_flag() -> None:
    """The declared director *is* the assertion `assume_gauge_carried` wanted."""
    tube = cxfc.SweptTube(
        static_helix, "km", tau_bounds=BOUNDS, director=lambda t: jnp.asarray(_E)
    )

    cxfc.rate_of_strain(tube, OFF_AXIS, u.Q(0.3, "s"))  # no flag, no raise


def test_a_bare_family_is_still_refused_off_axis() -> None:
    """Narrowing buys omission-raises; it does not loosen the old path."""
    family = lambda t: cxfc.TubularChart(
        cxfc.BishopBuilder(cxfc.AtTime(static_helix, t), "km"), tau_bounds=BOUNDS
    )

    with pytest.raises(RuntimeError, match="gauge-dependent"):
        cxfc.rate_of_strain(family, OFF_AXIS, u.Q(0.3, "s"))


def test_jit_and_vmap_over_a_scalar_time() -> None:
    """Criterion 6, which killed an earlier guard design."""
    tube = cxfc.SweptTube(
        static_helix, "km", tau_bounds=BOUNDS, director=lambda t: jnp.asarray(_E)
    )
    k = lambda tv: cxfc.rate_of_strain(tube, OFF_AXIS, u.Q(tv, "s")).value

    eager = np.asarray(k(0.3))
    assert np.allclose(np.asarray(jax.jit(k)(0.3)), eager)
    assert np.allclose(np.asarray(jax.vmap(k)(jnp.asarray([0.3, 0.4])))[0], eager)
