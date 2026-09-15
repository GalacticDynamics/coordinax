r"""A `Quantity` ``initial_normal`` must survive differentiation.

Only a *traced* one ever failed -- a constant `Quantity` has always worked.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc

BOUNDS = (u.Q(-1.0, "km"), u.Q(2.0, "km"))
POINT = {"tau": u.Q(0.5, "km"), "n1": u.Q(0.2, "km"), "n2": u.Q(0.1, "km")}
SPIN = 1.0 / np.sqrt(1.16)  # omega . T for this helix

T0 = jnp.asarray([0.0, 1.0, 0.4])
T0 = T0 / jnp.linalg.norm(T0)


def helix(sigma: u.AbstractQuantity, t: u.AbstractQuantity) -> u.AbstractQuantity:
    s = sigma.ustrip("km")
    return u.Q(jnp.stack([jnp.cos(s), jnp.sin(s), 0.4 * s]), "km")


def _spun(t: u.AbstractQuantity) -> jax.Array:
    """Rotated about ``T0`` -- depends on ``t``, so it is traced."""
    ang = SPIN * t.ustrip("s")
    c, s = jnp.cos(ang), jnp.sin(ang)
    e = jnp.asarray([1.0, 0.0, 0.0])
    return c * e + s * jnp.cross(T0, e) + (1 - c) * jnp.dot(T0, e) * T0


def _strain(wrap=lambda v: v, director=_spun) -> float:
    """`K_tau_tau` for ``director(t)`` wrapped by ``wrap``."""
    fam = lambda t: cxfc.TubularChart(
        cxfc.BishopBuilder(
            cxfc.AtTime(helix, t), "km", initial_normal=wrap(director(t))
        ),
        tau_bounds=BOUNDS,
    )
    # Off-axis, and legitimately so: `director(t)` is carried across the family
    # rather than each slice seeding itself from the world frame, which is
    # exactly what the opt-in asserts. See #870.
    return float(
        np.asarray(
            cxfc.rate_of_strain(
                fam, POINT, u.Q(0.0, "s"), assume_gauge_carried=True
            ).value
        )[0, 0]
    )


@pytest.fixture(scope="module")
def bare() -> float:
    """`_strain()` for a bare-array director -- the reference every test wants.

    It is a `jacfwd` through a diffrax solve, ~2.8s once the trace is
    compiled, and it is the same number every time. Module-scoped so a worker
    computes it once instead of six times.
    """
    return _strain()


def test_a_traced_quantity_director_matches_the_bare_array(bare: float) -> None:
    """The failing case: a `Quantity` director that depends on ``t``."""
    assert _strain(lambda v: u.Q(v, "")) == pytest.approx(bare, rel=1e-9)


@pytest.mark.parametrize("unit", ["", "km", "m", "pc"])
def test_the_director_s_unit_does_not_matter(unit: str, bare: float) -> None:
    """It is a direction: `_orthonormalize` discards the scale either way."""
    got = _strain(lambda v, _u=unit: u.Q(v, _u))
    assert got == pytest.approx(bare, rel=1e-9)


def test_the_value_is_the_one_only_a_director_can_produce(bare: float) -> None:
    """The curve has no ``t`` dependence, so only the director can move `K`."""
    fixed = lambda _t: jnp.asarray([1.0, 0.0, 0.0])
    assert _strain(director=fixed) == pytest.approx(0.0, abs=1e-12)
    assert bare == pytest.approx(-0.067526, abs=1e-5)
