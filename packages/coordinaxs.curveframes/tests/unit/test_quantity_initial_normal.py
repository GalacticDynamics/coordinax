r"""A `Quantity` ``initial_normal`` must survive differentiation.

`jnp.asarray` on a `Quantity` reaches for ``__array__``, which a *traced* one
cannot answer. So a director that depends on the differentiated argument --
which is the whole point of a director, and the natural typing for it -- raised
`TracerArrayConversionError` from inside `jax.jacfwd`, while the identical
direction as a bare array worked.

The vector is stripped in its *own* unit, so any unit is accepted: this is a
direction, `_orthonormalize` normalises it, and the scale is discarded anyway.
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
    """A helix with no ``t`` dependence at all."""
    s = sigma.ustrip("km")
    return u.Q(jnp.stack([jnp.cos(s), jnp.sin(s), 0.4 * s]), "km")


def _spun(t: u.AbstractQuantity) -> jax.Array:
    """Director rotated about ``T0``; depends on ``t``, so it is traced."""
    ang = SPIN * t.ustrip("s")
    c, s = jnp.cos(ang), jnp.sin(ang)
    e = jnp.asarray([1.0, 0.0, 0.0])
    return c * e + s * jnp.cross(T0, e) + (1 - c) * jnp.dot(T0, e) * T0


def _strain(wrap) -> float:
    """`K_tau_tau` for a director wrapped by ``wrap``."""
    fam = lambda t: cxfc.TubularChart(
        cxfc.BishopBuilder(cxfc.AtTime(helix, t), "km", initial_normal=wrap(_spun(t))),
        tau_bounds=BOUNDS,
    )
    return float(np.asarray(cxfc.rate_of_strain(fam, POINT, u.Q(0.0, "s")).value)[0, 0])


def test_a_traced_quantity_director_matches_the_bare_array() -> None:
    """The failing case: a `Quantity` director that depends on ``t``."""
    assert _strain(lambda v: u.Q(v, "")) == pytest.approx(
        _strain(lambda v: v), rel=1e-9
    )


@pytest.mark.parametrize("unit", ["", "km", "m", "pc"])
def test_the_director_s_unit_does_not_matter(unit: str) -> None:
    """It is a direction: `_orthonormalize` discards the scale either way."""
    got = _strain(lambda v, _u=unit: u.Q(v, _u))
    assert got == pytest.approx(_strain(lambda v: v), rel=1e-9)


def test_the_value_is_the_one_only_a_director_can_produce() -> None:
    """A static curve with a spinning director: no curve-derived rule can do this.

    The curve has no ``t`` dependence, so a gauge derived from it gives exactly
    zero; the spun director does not.
    """
    fixed = lambda t: jnp.asarray([1.0, 0.0, 0.0])
    fam = lambda t: cxfc.TubularChart(
        cxfc.BishopBuilder(cxfc.AtTime(helix, t), "km", initial_normal=fixed(t)),
        tau_bounds=BOUNDS,
    )
    still = float(
        np.asarray(cxfc.rate_of_strain(fam, POINT, u.Q(0.0, "s")).value)[0, 0]
    )
    assert still == pytest.approx(0.0, abs=1e-12)
    assert _strain(lambda v: v) == pytest.approx(-0.067526, abs=1e-5)
