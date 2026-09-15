r"""The same geometry must converge the same way in any ambient length unit.

`atol` was compared against two different kinds of quantity: a ``tau`` by the
bracketed solve, and the residual -- which is ``T . (x - gamma)``, a *length* --
by the unconstrained one. One scalar cannot be a tolerance on both, so the
identical circle converged differently depending on whether it was written in
km or in m.

Scaling the tolerances alone does not fix it: the sensitivity lives inside the
solvers' own convergence tests on the residual value. The residual itself has
to be unit-free.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc

#: Unit circle, probe at (1.5, 0.35); the nearest point is at this ``tau``.
TRUTH = float(np.arctan2(0.35, 1.5))
BOUNDS = (u.Q(0.0, "s"), u.Q(1.0, "s"))


def _circle(unit: str, scale: float):
    """The same geometry, written in ``unit`` at ``scale``."""

    def curve(tau: u.AbstractQuantity) -> u.AbstractQuantity:
        t = tau.ustrip("s")
        return u.Q(scale * jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), unit)

    return curve


def _solve(unit: str, scale: float) -> float:
    builder = cxfc.FrenetSerretBuilder(_circle(unit, scale), "s")
    x = u.Q(scale * jnp.asarray([1.5, 0.35, 0.0]), unit)
    return float(cxfc.nearest_tau(builder, x, bounds=BOUNDS).ustrip("s"))


#: One circle of radius 1 km, written four ways. The scales are the actual
#: conversions, so each case really is the same geometry -- ``pc`` at 1.0 was a
#: *different* circle and exercised no conversion factor at all.
_KM_IN_PC = float(u.Q(1.0, "km").ustrip("pc"))  # from unxt, not hard-coded

#: Solved once, not once per case: it is the same baseline every time.
_km_answer = _solve("km", 1.0)


@pytest.mark.parametrize(
    ("unit", "scale"), [("km", 1.0), ("m", 1000.0), ("Mm", 0.001), ("pc", _KM_IN_PC)]
)
def test_the_answer_does_not_depend_on_the_ambient_unit(
    unit: str, scale: float
) -> None:
    """Every spelling of one circle must give one answer."""
    assert _solve(unit, scale) == pytest.approx(_km_answer, rel=1e-12)


def test_and_that_answer_is_right() -> None:
    """Unit-invariance would also be satisfied by being uniformly wrong."""
    assert _km_answer == pytest.approx(TRUTH, abs=1e-7)
