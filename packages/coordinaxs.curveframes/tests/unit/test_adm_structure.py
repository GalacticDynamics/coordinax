r"""The worldtube's metric is already an ADM decomposition.

Nobody built it that way. `metric_matrix` falls through to the generic
Jacobian pullback, and on a pinned-station chart -- whose coordinates are
$(t, n_1, n_2)$, one time and two lengths -- that pullback produces the ADM
block structure on its own:

$$ g_{00} = |\boldsymbol\beta + \dot R\,\mathbf{n}|^2, \qquad
   g_{0i} = (R\boldsymbol\beta)_i, \qquad
   g_{ij} = \delta_{ij}. $$

Emergent structure is the kind that breaks silently: one refactor of the
pullback and $g_{0i}$ stops being the shift, with nothing to notice. These
pin it.

Galilean spacetime has no non-degenerate 4-metric -- a temporal 1-form and a
spatial 3-metric, not a single $g_{\mu\nu}$ -- so this is a 3-metric on one
section, not a 4-metric. `(gamma_ij, beta^i)` is the container; see #779.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from coordinaxs.api.manifolds import metric_matrix

import unxt as u

import coordinaxs.curveframes as cxfc

S0 = 1.3
T0 = 1.0


def stretch_and_bend(
    sigma: u.AbstractQuantity, t: u.AbstractQuantity
) -> u.AbstractQuantity:
    """The #778 curve: stretches *and* bends, so the triad genuinely rotates."""
    sv, tv = sigma.ustrip("km"), t.ustrip("s")
    x = sv * (1.0 + 0.5 * tv)
    y = 0.1 * tv * sv**2
    return u.Q(jnp.stack([x, y, jnp.zeros_like(x)]), "km")


def _worldtube():
    b = cxfc.BishopBuilder(stretch_and_bend, "km", station=u.Q(S0, "km"))
    ch = cxfc.TubularChart(b, tau_bounds=(u.Q(0.0, "s"), u.Q(2.0, "s")))
    return b, ch


def _g(ch, n1: float, n2: float) -> np.ndarray:
    pt = {"tau": u.Q(T0, "s"), "n1": u.Q(n1, "km"), "n2": u.Q(n2, "km")}
    return np.asarray(metric_matrix(ch.M, pt, ch).matrix.value)


def test_the_blocks_carry_the_adm_units() -> None:
    """A speed squared, a speed, and a dimensionless spatial block."""
    _, ch = _worldtube()
    pt = {"tau": u.Q(T0, "s"), "n1": u.Q(0.2, "km"), "n2": u.Q(0.1, "km")}
    unit_str = str(metric_matrix(ch.M, pt, ch).matrix.unit)

    assert "km2 / s2" in unit_str  # g_00, a speed squared
    assert "km / s" in unit_str  # g_0i, a speed


def test_the_off_diagonal_block_is_the_shift() -> None:
    r"""$g_{0i}$ *is* $\boldsymbol\beta$, in the triad's normal directions.

    Not approximately, and not by construction -- the pullback and
    `velocity` reach it by different routes and agree.
    """
    b, ch = _worldtube()
    beta = np.asarray(b.velocity(u.Q(T0, "s")).ustrip("km/s"))
    rot = np.asarray(b.rotation_matrix(u.Q(T0, "s")))
    in_triad = rot @ beta  # (T, U1, U2) components

    g = _g(ch, 0.2, 0.1)
    assert np.allclose([g[0, 1], g[0, 2]], in_triad[1:], atol=1e-5)
    assert np.allclose(g[0, 1], g[1, 0], atol=1e-6)  # symmetric


def test_the_spatial_block_is_the_orthonormal_triad() -> None:
    """`U1` and `U2` are unit and orthogonal, so the normal block is the identity."""
    _, ch = _worldtube()
    g = _g(ch, 0.4, -0.3)
    assert np.allclose(g[1:, 1:], np.eye(2), atol=1e-5)


def test_g00_is_the_squared_speed_of_the_point_at_that_offset() -> None:
    r"""On the axis it is $|\boldsymbol\beta|^2$; off it, $\dot R\,\mathbf n$ enters.

    The same $\dot R\,\mathbf n$ term that makes a closed-form
    $R(v-\boldsymbol\beta)$ wrong off-axis (see `velocity`'s notes). Here the
    pullback carries it correctly, which is worth pinning rather than
    trusting.
    """
    b, ch = _worldtube()
    beta = np.asarray(b.velocity(u.Q(T0, "s")).ustrip("km/s"))
    drot = np.asarray(jax.jacfwd(lambda tv: b.rotation_matrix(u.Q(tv, "s")))(T0))

    assert np.allclose(_g(ch, 0.0, 0.0)[0, 0], beta @ beta, atol=1e-5)

    for n1, n2 in [(0.2, 0.0), (0.5, 0.3), (1.0, -0.4)]:
        speed = beta + n1 * drot[1] + n2 * drot[2]
        assert np.allclose(_g(ch, n1, n2)[0, 0], speed @ speed, atol=1e-4)


@pytest.mark.parametrize("n1", [0.0, 0.5], ids=["on-axis", "off-axis"])
def test_the_spatial_slice_is_three_lengths(n1: float) -> None:
    """`AtTime` gives the other section: no time coordinate, so no shift block."""
    ch = cxfc.TubularChart(
        cxfc.BishopBuilder(cxfc.AtTime(stretch_and_bend, u.Q(T0, "s")), "km"),
        tau_bounds=(u.Q(0.0, "km"), u.Q(3.0, "km")),
    )
    assert ch.coord_dimensions == ("length", "length", "length")

    pt = {"tau": u.Q(S0, "km"), "n1": u.Q(n1, "km"), "n2": u.Q(0.0, "km")}
    g = metric_matrix(ch.M, pt, ch)
    # three length coordinates in one unit, so every entry is dimensionless
    assert "km" not in str(g.matrix.unit)
    assert np.allclose(np.asarray(g.matrix.value)[1:, 1:], np.eye(2), atol=1e-5)
