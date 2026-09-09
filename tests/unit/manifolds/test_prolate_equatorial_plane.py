r"""The prolate-spheroidal metric must survive the ``nu = 0`` plane.

``nu = 0`` is the equatorial disc -- ``z = 0``, the galactic plane, and the
commonest case this chart is used for. ``check_data`` admits it (the domain is
``mu >= Delta^2``, ``|nu| <= Delta^2``) and ``pt_map`` round-trips through it
exactly, but the metric came back all-NaN.

Only ``g_nu_nu`` is genuinely singular there: ``nu`` is a degenerate coordinate
on the disc. ``g_mu_mu`` and ``g_phi_phi`` are finite and well defined, and were
being lost with it.

The cause is not the shape of the expression -- factoring ``sqrt(mu * nu_D2)``
into a product of roots does not help, and was tried. Forward-mode AD evaluates
``d(sqrt(t))`` as ``0.5/sqrt(t) * tangent``, so at ``t = 0`` *every* column of
the pullback picks up ``inf * 0``: the whole ``dz`` row came back NaN,
``dz/dphi`` included, which is identically zero.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm

CHART = cxc.ProlateSpheroidal3D(Delta=u.Q(2.0, "kpc"))
ON_PLANE = {"x": u.Q(3.0, "kpc"), "y": u.Q(4.0, "kpc"), "z": u.Q(0.0, "kpc")}


def _diag_at(nu_value: float) -> np.ndarray:
    """Metric diagonal at the in-plane point, with ``nu`` overridden."""
    p = dict(cxc.pt_map(ON_PLANE, cxc.cart3d, CHART))
    p["nu"] = u.Q(nu_value, "kpc2")
    return np.asarray(cxm.metric_matrix(cxm.R3, p, CHART).diagonal.value)


def test_the_in_plane_point_is_admitted_and_round_trips() -> None:
    """Establishes that this is an interior point, not an invalid query."""
    p = dict(cxc.pt_map(ON_PLANE, cxc.cart3d, CHART))
    assert float(p["nu"].value) == 0.0
    CHART.check_data(p)  # must not raise
    back = cxc.pt_map(p, CHART, cxc.cart3d)
    assert float(back["x"].ustrip("kpc")) == pytest.approx(3.0, abs=1e-9)


def test_the_finite_components_are_finite_on_the_plane() -> None:
    """``g_mu_mu`` and ``g_phi_phi`` have limits; they must not be NaN."""
    got = _diag_at(0.0)
    assert np.isfinite(got[0]), f"g_mu_mu = {got[0]}"
    assert np.isfinite(got[2]), f"g_phi_phi = {got[2]}"


def test_they_match_the_limit_approached_from_off_plane() -> None:
    """The value on the plane must agree with the limit from just off it."""
    near = _diag_at(1e-10)
    got = _diag_at(0.0)
    assert got[0] == pytest.approx(near[0], rel=1e-6)
    assert got[2] == pytest.approx(near[2], rel=1e-6)
    # And those limits are the analytic ones for this point.
    assert got[0] == pytest.approx(0.01, rel=1e-6)
    assert got[2] == pytest.approx(25.0, rel=1e-6)


def test_the_degenerate_component_is_still_reported_as_singular() -> None:
    """``nu`` really is degenerate on the disc -- that must not be hidden.

    Asserts ``isinf`` rather than ``not isfinite``: NaN satisfies the latter,
    so the weaker form passed on the very bug this file is about.
    """
    got = _diag_at(0.0)
    assert np.isinf(got[1]), f"g_nu_nu = {got[1]}, expected +inf"


@pytest.mark.parametrize("mu", [5.0, 29.0, 400.0])
@pytest.mark.parametrize("nu", [-3.5, -0.7, 0.3, 2.0, 3.9])
def test_it_agrees_with_the_jacobian_pullback_off_the_plane(
    mu: float, nu: float
) -> None:
    """Away from the degeneracy the pullback works, and is the oracle.

    The closed form replaced it, so it must reproduce it exactly wherever the
    pullback is defined -- both signs of ``nu``, approaching both domain edges.
    """
    p = {"mu": u.Q(mu, "kpc2"), "nu": u.Q(nu, "kpc2"), "phi": u.Angle(0.7, "rad")}
    closed = np.asarray(cxm.metric_matrix(cxm.R3, p, CHART).diagonal.value)
    jac = cxc.jac_pt_map(p, CHART, cxc.cart3d)
    pullback = np.asarray((jac.T @ jac).value).diagonal()
    assert np.allclose(closed, pullback, rtol=1e-12, atol=0.0), (
        f"closed {closed} vs pullback {pullback}"
    )


def test_the_angular_unit_follows_the_plain_array_convention() -> None:
    """A `Quantity` angle gives ``/ rad2``; a plain array stays dimensionless.

    This is the convention `_angle_basis_unit` carries for every other rule in
    the module (#835). Hard-coding ``rad`` here would have reported ``/ rad2``
    to a caller who passed bare arrays.
    """
    base = {"mu": u.Q(29.0, "kpc2"), "nu": u.Q(1.0, "kpc2")}
    with_qty = cxm.metric_matrix(cxm.R3, {**base, "phi": u.Angle(0.7, "rad")}, CHART)
    with_bare = cxm.metric_matrix(cxm.R3, {**base, "phi": jnp.asarray(0.7)}, CHART)

    assert "rad2" in str(with_qty.diagonal.unit)
    assert "rad" not in str(with_bare.diagonal.unit)
    # The values are the same either way -- only the label differs.
    assert np.allclose(
        np.asarray(with_qty.diagonal.value), np.asarray(with_bare.diagonal.value)
    )
