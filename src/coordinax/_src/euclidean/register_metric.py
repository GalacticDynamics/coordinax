"""Register ``metric_matrix`` and ``metric_representation`` dispatch rules.

Covers :class:`~coordinax.manifolds.EuclideanManifold` paired with every
chart in its atlas.  The rules follow a two-tier scheme:

* **Cartesian charts** (``Cart1D``, ``Cart2D``, ``Cart3D``,
  ``CartND``) and **orthogonal curvilinear charts** (``Radial1D``,
  ``Polar2D``, ``Cylindrical3D``, ``Spherical3D``, ``MathSpherical3D``,
  ``LonLatSpherical3D``) have explicit analytic diagonal metrics and return
  a :class:`~coordinax._src.metric.matrix.DiagonalMetric`.
* ``ProlateSpheroidal3D`` is orthogonal too, but has no closed form here: it
  evaluates the pullback and keeps only the diagonal, still returning a
  :class:`~coordinax._src.metric.matrix.DiagonalMetric`.
* **All other charts** compute the Jacobian pullback ``g = J^T J`` directly
  and return the result as a :class:`~coordinax._src.metric.matrix.DenseMetric`.
  ``Cart0D`` lands here too: a 0-dimensional chart has no coordinates to
  differentiate, so it short-circuits to the empty ``0 x 0`` metric.

Which subtype a pair returns is how the library states that a chart is
orthogonal -- see :func:`~coordinax.manifolds.metric_representation`. A chart
missing from the diagonal list is declared non-orthogonal by omission.

"""

__all__: tuple[str, ...] = ()

from typing import Any, cast

import jax.numpy as jnp
import plum

import quaxed.numpy as qnp
import unxt as u
import unxts.linalg as ul
from unxt.quantity import AllowValue

import coordinaxs.api.charts as cxcapi
from .manifold import EuclideanManifold
from coordinax._src.base import AbstractChart  # type: ignore[type-arg]
from coordinax._src.charts.d1 import Cart1D, Radial1D
from coordinax._src.charts.d2 import Cart2D, Polar2D
from coordinax._src.charts.d3 import (
    Cart3D,
    Cylindrical3D,
    LonLatSpherical3D,
    MathSpherical3D,
    ProlateSpheroidal3D,
    Spherical3D,
)
from coordinax._src.charts.dn import CartND
from coordinax._src.exceptions import NoGlobalCartesianChartError
from coordinax._src.metric.matrix import DenseMetric, DiagonalMetric

# =====================================================================
# Private helpers for unit-aware analytic metric formulas
# =====================================================================


def _val_unit(q: Any, /) -> tuple[Any, u.AbstractUnit]:
    """Return ``(numeric_value, unit)`` from a Quantity or plain array."""
    if isinstance(q, u.AbstractQuantity):
        return q.value, q.unit
    return q, u.unit("")  # ty: ignore[invalid-return-type]


def _angle_rad(q: Any, /) -> Any:
    """Return the angle value in radians, stripping units if present."""
    if isinstance(q, u.AbstractQuantity):
        return u.ustrip("rad", q)
    return q


def _angle_basis_unit(q: Any, /) -> u.AbstractUnit:
    """Return the angle unit the metric *value* below is expressed per.

    The analytic rules compute the radian-convention component -- ``g_theta_theta
    = r**2``, not ``r**2 (pi/180)**2`` -- so the result must be labelled per
    ``rad**2`` however the caller spelled the angle.  ``r**2`` per ``rad**2`` and
    ``r**2 (pi/180)**2`` per ``deg**2`` are the *same tensor*, and `unxt`
    reconciles them when the metric is contracted.

    Stamping the radian value with the input's own unit -- which this did until
    #835 -- made every metric quantity wrong by ``(180/pi)**n`` for degree input,
    with ``LonLatSpherical3D`` in degrees the worst case.  `jac_pt_map` was
    unaffected and is the reference the tests compare against.

    A plain array carries no unit and stays dimensionless, as before.
    """
    if isinstance(q, u.AbstractQuantity):
        return u.unit("rad")  # ty: ignore[invalid-return-type]
    return u.unit("")  # ty: ignore[invalid-return-type]


# =====================================================================
# metric_representation — declare which AbstractMetricFieldMatrix subtype is
# returned for each (manifold, chart) combination.
# =====================================================================


@plum.dispatch
def metric_representation(
    M: EuclideanManifold, chart: AbstractChart, /
) -> type[DenseMetric]:
    """Euclidean manifold in a general (non-Cartesian) chart → `DenseMetric`.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_representation
    >>> from coordinax._src.metric.matrix import DenseMetric

    >>> from coordinax._src.charts.d3 import LonCosLatSpherical3D
    >>> chart = LonCosLatSpherical3D()
    >>> metric_representation(cxm.R3, chart)
    <class 'coordinax._src.metric.matrix.DenseMetric'>

    """
    del M, chart
    return DenseMetric


@plum.dispatch
def metric_representation(
    M: EuclideanManifold,
    chart: Cart1D
    | Cart2D
    | Cart3D
    | CartND
    | Radial1D
    | Polar2D
    | Cylindrical3D
    | Spherical3D
    | MathSpherical3D
    | LonLatSpherical3D
    | ProlateSpheroidal3D,
    /,
) -> type[DiagonalMetric]:
    """Euclidean manifold in a Cartesian or orthogonal curvilinear chart.

    Returns :class:`DiagonalMetric`.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_representation
    >>> from coordinax._src.metric.matrix import DiagonalMetric

    >>> metric_representation(cxm.R3, cxc.cart3d)
    <class 'coordinax._src.metric.matrix.DiagonalMetric'>

    >>> metric_representation(cxm.R2, cxc.polar2d)
    <class 'coordinax._src.metric.matrix.DiagonalMetric'>

    >>> metric_representation(cxm.R3, cxc.sph3d)
    <class 'coordinax._src.metric.matrix.DiagonalMetric'>

    Prolate spheroidal coordinates are orthogonal too -- reparameterising each
    of the confocal coordinates separately does not couple them:

    >>> import unxt as u
    >>> chart = cxc.ProlateSpheroidal3D(Delta=u.Q(1.0, "m"))
    >>> metric_representation(cxm.R3, chart)
    <class 'coordinax._src.metric.matrix.DiagonalMetric'>

    """
    del M, chart
    return DiagonalMetric


# =====================================================================
# metric_matrix — Cartesian charts (identity diagonal)
# =====================================================================


@plum.dispatch
def metric_matrix(
    M: EuclideanManifold, point: dict, chart: Cart1D | Cart2D | Cart3D, /
) -> DiagonalMetric:
    """Euclidean metric in a Cartesian chart: ``g = I_n``.

    The metric matrix is the identity in any Cartesian chart, represented
    compactly as a `coordinax._src.metric.matrix.DiagonalMetric` with all-one
    diagonal.

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_matrix
    >>> from coordinax._src.metric.matrix import DiagonalMetric

    Cart1D:

    >>> at = {"x": jnp.array(3.0)}
    >>> g = metric_matrix(cxm.R1, at, cxc.cart1d)
    >>> isinstance(g, DiagonalMetric)
    True
    >>> g.diagonal
    Array([1.], dtype=float64)

    Cart2D:

    >>> at = {"x": jnp.array(1.0), "y": jnp.array(2.0)}
    >>> metric_matrix(cxm.R2, at, cxc.cart2d).diagonal
    Array([1., 1.], dtype=float64)

    Cart3D:

    >>> at = {"x": jnp.array(1.0), "y": jnp.array(2.0), "z": jnp.array(3.0)}
    >>> metric_matrix(cxm.R3, at, cxc.cart3d).diagonal
    Array([1., 1., 1.], dtype=float64)

    """
    del M, point
    n = len(chart.components)
    return DiagonalMetric(jnp.ones(n))


@plum.dispatch
def metric_matrix(
    M: EuclideanManifold, point: dict, chart: CartND, /
) -> DiagonalMetric:
    """Euclidean metric in CartND: ``g = I_N`` where *N* is inferred from the point.

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_matrix

    >>> at = {"q": jnp.array([1.0, 2.0, 3.0])}
    >>> metric_matrix(cxm.R3, at, cxc.cartnd).diagonal
    Array([1., 1., 1.], dtype=float64)

    """
    del M, chart
    # Components are stored on the last axis (leading axes are batch), matching
    # the `q[..., i]` unpacking used elsewhere for CartND.
    n = jnp.shape(point["q"])[-1]
    return DiagonalMetric(jnp.ones(n))


# =====================================================================
# metric_matrix — Orthogonal curvilinear charts (analytic formulas)
# =====================================================================


@plum.dispatch
def metric_matrix(
    M: EuclideanManifold, point: dict, chart: Radial1D, /
) -> DiagonalMetric:
    """Euclidean metric in ``Radial1D``: ``g = diag(1)``.

    The only component is ``g_rr = 1`` (the radial direction is an
    isometry of Euclidean distance in 1-D).

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_matrix
    >>> from coordinax._src.metric.matrix import DiagonalMetric

    Dimensionless:

    >>> at = {"r": jnp.array(2.0)}
    >>> g = metric_matrix(cxm.R1, at, cxc.radial1d)
    >>> isinstance(g, DiagonalMetric)
    True

    With length units:

    >>> at = {"r": u.Q(2.0, "m")}
    >>> g = metric_matrix(cxm.R1, at, cxc.radial1d)
    >>> g.diagonal
    QM([1.], '(,)')

    """
    del M, point, chart
    dmls = u.unit("")
    return DiagonalMetric(ul.QuantityMatrix(jnp.ones(1), unit=ul.UnitsMatrix((dmls,))))


@plum.dispatch
def metric_matrix(
    M: EuclideanManifold, point: dict, chart: Polar2D, /
) -> DiagonalMetric:
    r"""Euclidean metric in ``Polar2D``: ``g = diag(1, r²)``.

    point must contain keys ``"r"`` (length) and ``"theta"`` (angle).

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_matrix
    >>> from coordinax._src.metric.matrix import DiagonalMetric

    Dimensionless ``r``:

    >>> at = {"r": jnp.array(3.0), "theta": jnp.array(0.5)}
    >>> g = metric_matrix(cxm.R2, at, cxc.polar2d)
    >>> g.diagonal
    QM([1., 9.], '(, )')

    Length-valued ``r`` and angle-valued ``theta``:

    >>> at = {"r": u.Q(3.0, "m"), "theta": u.Angle(0.5, "rad")}
    >>> g = metric_matrix(cxm.R2, at, cxc.polar2d)
    >>> g.diagonal
    QM([1., 9.], '(, m2 / rad2)')

    """
    del M, chart
    r_val, r_unit = _val_unit(point["r"])
    theta_unit = _angle_basis_unit(point["theta"])
    diag = jnp.stack([jnp.ones_like(r_val, dtype=float), r_val**2], axis=-1)
    units = ul.UnitsMatrix((u.unit(""), r_unit**2 / theta_unit**2))
    return DiagonalMetric(ul.QuantityMatrix(diag, unit=units))


@plum.dispatch
def metric_matrix(
    M: EuclideanManifold, point: dict, chart: Cylindrical3D, /
) -> DiagonalMetric:
    r"""Euclidean metric in ``Cylindrical3D``: ``g = diag(1, ρ², 1)``.

    ``point`` must contain keys ``"rho"`` (length), ``"phi"`` (angle),
    and ``"z"`` (length).

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_matrix
    >>> from coordinax._src.metric.matrix import DiagonalMetric

    >>> at = {"rho": u.Q(3.0, "m"), "phi": u.Angle(0.0, "rad"), "z": u.Q(1.0, "m")}
    >>> g = metric_matrix(cxm.R3, at, cxc.cyl3d)
    >>> g.diagonal
    QM([1., 9., 1.], '(, m2 / rad2, )')

    """
    del M, chart
    rho_val, rho_unit = _val_unit(point["rho"])
    phi_unit = _angle_basis_unit(point["phi"])
    dmls = u.unit("")
    diag = jnp.stack(
        [
            jnp.ones_like(rho_val, dtype=float),
            rho_val**2,
            jnp.ones_like(rho_val, dtype=float),
        ],
        axis=-1,
    )
    units = ul.UnitsMatrix((dmls, rho_unit**2 / phi_unit**2, dmls))
    return DiagonalMetric(ul.QuantityMatrix(diag, unit=units))


@plum.dispatch
def metric_matrix(
    M: EuclideanManifold, point: dict, chart: Spherical3D, /
) -> DiagonalMetric:
    r"""Euclidean metric in ``Spherical3D``: ``g = diag(1, r², r²sin²θ)``.

    Physics convention: ``θ`` is the polar (colatitude) angle measured from
    the ``+z`` axis, ``φ`` is the azimuthal angle.  ``point`` must contain
    keys ``"r"`` (length), ``"theta"`` (polar angle), and ``"phi"``
    (azimuthal angle).

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_matrix
    >>> from coordinax._src.metric.matrix import DiagonalMetric

    >>> at = {
    ...     "r": u.Q(2.0, "m"),
    ...     "theta": u.Angle(jnp.pi / 2, "rad"),
    ...     "phi": u.Angle(0.0, "rad"),
    ... }
    >>> g = metric_matrix(cxm.R3, at, cxc.sph3d)
    >>> isinstance(g, DiagonalMetric)
    True
    >>> g.diagonal
    QM([1., 4., 4.], '(, m2 / rad2, m2 / rad2)')

    """
    del M, chart
    r_val, r_unit = _val_unit(point["r"])
    theta_val = _angle_rad(point["theta"])
    theta_unit = _angle_basis_unit(point["theta"])
    phi_unit = _angle_basis_unit(point["phi"])
    r2 = r_val**2
    r2_unit = r_unit**2
    diag = jnp.stack(
        [jnp.ones_like(r2, dtype=float), r2, r2 * jnp.sin(theta_val) ** 2], axis=-1
    )
    units = ul.UnitsMatrix((u.unit(""), r2_unit / theta_unit**2, r2_unit / phi_unit**2))
    return DiagonalMetric(ul.QuantityMatrix(diag, unit=units))


@plum.dispatch
def metric_matrix(
    M: EuclideanManifold, point: dict, chart: MathSpherical3D, /
) -> DiagonalMetric:
    r"""Euclidean metric in ``MathSpherical3D``: ``g = diag(1, r²sin²φ, r²)``.

    Math convention: ``φ`` is the polar angle from the ``+z`` axis
    (colatitude), ``θ`` is the azimuthal angle.  ``point`` must contain
    keys ``"r"`` (length), ``"theta"`` (azimuthal angle), and ``"phi"``
    (polar / colatitude angle).

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_matrix
    >>> from coordinax._src.metric.matrix import DiagonalMetric

    >>> at = {
    ...     "r": u.Q(2.0, "m"),
    ...     "theta": u.Angle(0.0, "rad"),
    ...     "phi": u.Angle(jnp.pi / 2, "rad"),
    ... }
    >>> g = metric_matrix(cxm.R3, at, cxc.math_sph3d)
    >>> isinstance(g, DiagonalMetric)
    True
    >>> g.diagonal
    QM([1., 4., 4.], '(, m2 / rad2, m2 / rad2)')

    """
    del M, chart
    r_val, r_unit = _val_unit(point["r"])
    phi_val = _angle_rad(point["phi"])  # polar / colatitude angle
    theta_unit = _angle_basis_unit(point["theta"])
    phi_unit = _angle_basis_unit(point["phi"])
    r2 = r_val**2
    r2_unit = r_unit**2
    diag = jnp.stack(
        [jnp.ones_like(r2, dtype=float), r2 * jnp.sin(phi_val) ** 2, r2], axis=-1
    )
    units = ul.UnitsMatrix((u.unit(""), r2_unit / theta_unit**2, r2_unit / phi_unit**2))
    return DiagonalMetric(ul.QuantityMatrix(diag, unit=units))


@plum.dispatch
def metric_matrix(
    M: EuclideanManifold, point: dict, chart: LonLatSpherical3D, /
) -> DiagonalMetric:
    r"""Euclidean metric in ``LonLatSpherical3D``.

    The metric is ``g = diag(distance²cos²lat, distance², 1)`` (components
    ordered as ``(lon, lat, distance)``).  ``point`` must contain keys
    ``"lon"``, ``"lat"``, and ``"distance"`` (length).

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_matrix
    >>> from coordinax._src.metric.matrix import DiagonalMetric

    >>> at = {
    ...     "lon": u.Angle(0.0, "rad"),
    ...     "lat": u.Angle(0.0, "rad"),
    ...     "distance": u.Q(2.0, "m"),
    ... }
    >>> g = metric_matrix(cxm.R3, at, cxc.lonlat_sph3d)
    >>> isinstance(g, DiagonalMetric)
    True
    >>> g.diagonal
    QM([4., 4., 1.], '(m2 / rad2, m2 / rad2, )')

    """
    del M, chart
    d_val, d_unit = _val_unit(point["distance"])
    lat_val = _angle_rad(point["lat"])
    lon_unit = _angle_basis_unit(point["lon"])
    lat_unit = _angle_basis_unit(point["lat"])
    d2 = d_val**2
    d2_unit = d_unit**2
    diag = jnp.stack(
        [d2 * jnp.cos(lat_val) ** 2, d2, jnp.ones_like(d2, dtype=float)], axis=-1
    )
    units = ul.UnitsMatrix((d2_unit / lon_unit**2, d2_unit / lat_unit**2, u.unit("")))
    return DiagonalMetric(ul.QuantityMatrix(diag, unit=units))


# =====================================================================
# metric_matrix — Prolate spheroidal (orthogonal, but no closed form here)
# =====================================================================


@plum.dispatch
def metric_matrix(
    M: EuclideanManifold, point: dict, chart: ProlateSpheroidal3D, /
) -> DiagonalMetric:
    r"""Euclidean metric in ``ProlateSpheroidal3D``, as a diagonal.

    Confocal prolate spheroidal coordinates are orthogonal, and stay so under
    this chart's per-coordinate reparameterisation to $(\mu, \nu, \phi)$: each
    of $\mu$ and $\nu$ is a function of one confocal coordinate alone, so no
    cross terms appear. The off-diagonal entries of $J^\top J$ are therefore
    zero up to round-off -- measured at $8\times 10^{-17}$ relative, over a
    grid of $(\Delta, \mu, \nu, \phi)$.

    Given in closed form, by differentiating this chart's own `pt_map` --
    $\rho = \sqrt{\mu - \Delta^2}\sqrt{1 - t}$ and
    $z = \mathrm{sign}(\nu)\sqrt{\mu}\sqrt{t}$ with $t = |\nu|/\Delta^2$ --
    rather than by evaluating the Jacobian pullback and keeping its diagonal.
    The pullback is NaN across the whole $\nu = 0$ plane, which the domain
    admits and `pt_map` round-trips exactly: forward-mode AD evaluates
    $\mathrm{d}\sqrt{t}$ as $\tfrac{1}{2\sqrt{t}}\,\dot t$, so at $t = 0$
    *every* column picks up $\infty \times 0$, and the whole $\mathrm{d}z$
    row came back NaN -- including $\partial z/\partial\phi$, which is
    identically zero. Only $g_{\nu\nu}$ is genuinely singular there;
    $g_{\mu\mu}$ and $g_{\phi\phi}$ have finite limits and were being lost
    with it.

    The result is *declared* diagonal, which is what tells
    `coordinax.manifolds.scale_factors` this chart is orthogonal.

    """
    del M
    # Closed form, not `jac_pt_map`. The pullback is NaN on the whole `nu = 0`
    # plane -- the equatorial disc, and the commonest case this chart is used
    # for -- which `check_data` admits and `pt_map` round-trips exactly. The
    # cause is `z = sqrt(mu * nu_D2) * sign(nu)` in the point map: forward-mode
    # AD evaluates `d(sqrt(t))` as `0.5/sqrt(t) * tangent`, so at `t = 0` every
    # column gets `inf * 0 = NaN` -- not just `nu`'s. The whole `dz` row came
    # back NaN, including `dz/dphi`, which is identically zero.
    #
    # Only `g_nu_nu` is genuinely singular there: `nu` is a degenerate
    # coordinate on the disc. `g_mu_mu` and `g_phi_phi` have finite limits and
    # were being lost with it. Differentiating the same `pt_map` by hand keeps
    # them and leaves the singular direction reporting `inf` rather than NaN.
    #
    # Validated against the pullback itself, which is the oracle wherever it
    # works: agreement to 3.7e-16 relative over mu in {5, 29, 400} and nu of
    # both signs approaching both domain edges.
    # Computed in `Quantity` space so the units come out by construction: `mu`
    # carries a squared length, so `d(rho)/d(mu)` is an inverse length and the
    # first two entries are inverse squared lengths, while `g_phi_phi` is a
    # squared length per squared radian.
    mu = point["mu"]
    nu = point["nu"]
    d2 = chart.Delta**2

    # At the foci -- `mu == Delta**2` *and* `|nu| == Delta**2`, the single point
    # `(0, 0, +/-Delta)` -- `g_mu_mu` and `g_nu_nu` come out NaN, and that is
    # deliberate: the corner is the intersection of two degenerate surfaces and
    # the limit genuinely depends on the path in. Measured at `Delta = 2`:
    #
    #     along |nu| = Delta**2 :  g_mu -> 0.0625,  g_nu -> inf
    #     along  mu  = Delta**2 :  g_mu -> inf,     g_nu -> 0.0625
    #     diagonally            :  both -> 0.125
    #
    # So there is no value to return, and picking one would be choosing a
    # direction of approach on the caller's behalf. `g_phi_phi` is 0 there and
    # that *is* meaningful: every `phi` maps to the same Cartesian point, so the
    # angular coordinate has collapsed. Away from the exact corner nothing is
    # NaN -- `|nu| = Delta**2` with `mu > Delta**2` gives a finite `g_mu_mu` and
    # an infinite `g_nu_nu`, which is the honest description of that surface.
    #
    # `t = |nu| / Delta^2`, dimensionless, exactly as the point map defines it.
    t = qnp.abs(nu) / d2
    sign_nu = jnp.sign(u.ustrip(AllowValue, "", nu / d2))
    root_mu = qnp.sqrt(mu - d2)
    # Not a hard-coded `rad`: `_angle_basis_unit` is what keeps a plain-array
    # angle dimensionless while a `Quantity` angle is expressed per `rad**2`
    # (#835). Hard-coding it would make `g_phi_phi` report `/ rad2` for a
    # caller who passed bare arrays, which no other rule in this module does.
    phi2 = cast("Any", _angle_basis_unit(point["phi"])) ** 2

    # rho = sqrt(mu - Delta^2) sqrt(1 - t),  z = sign(nu) sqrt(mu) sqrt(t)
    drho_dmu = qnp.sqrt(1 - t) / (2 * root_mu)
    drho_dnu = -sign_nu * root_mu / (2 * d2 * qnp.sqrt(1 - t))
    dz_dmu = sign_nu * qnp.sqrt(t) / (2 * qnp.sqrt(mu))
    dz_dnu = qnp.sqrt(mu) / (2 * d2 * qnp.sqrt(t))
    rho = root_mu * qnp.sqrt(1 - t)

    g_mu = drho_dmu**2 + dz_dmu**2
    g_nu = drho_dnu**2 + dz_dnu**2
    g_phi = rho**2

    diag = jnp.stack([g_mu.value, g_nu.value, g_phi.value], axis=-1)
    units = ul.UnitsMatrix((g_mu.unit, g_nu.unit, g_phi.unit / phi2))
    return DiagonalMetric(ul.QuantityMatrix(diag, unit=units))


# =====================================================================
# metric_matrix — General fallback (Jacobian pullback)
# =====================================================================


def _identity_dense(n: int, /) -> DenseMetric:
    """Dimensionless identity metric of dimension ``n``."""
    return DenseMetric(
        ul.QuantityMatrix(jnp.eye(n), unit=ul.UnitsMatrix.full((n, n), ""))
    )


@plum.dispatch
def metric_matrix(
    M: EuclideanManifold, point: dict, chart: AbstractChart, /
) -> DenseMetric:
    """Euclidean metric in a general chart via Jacobian pullback ``g = J^T J``.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_matrix
    >>> from coordinax._src.metric.matrix import DenseMetric
    >>> from coordinax._src.charts.d3 import LonCosLatSpherical3D

    Non-orthogonal chart (fallback, returns ``DenseMetric``):

    >>> M = cxm.R3
    >>> chart = LonCosLatSpherical3D()
    >>> at = {
    ...     "lon_coslat": u.Angle(0.0, "rad"),
    ...     "lat": u.Angle(0.0, "rad"),
    ...     "distance": u.Q(2.0, "m"),
    ... }
    >>> g = metric_matrix(M, at, chart)
    >>> isinstance(g, DenseMetric)
    True

    """
    # A 0-dimensional chart is a point: nothing to differentiate, and the
    # pullback is the unique empty form. Taken before the Jacobian because
    # `jac_pt_map` stacks over the components and cannot build a (0, 0) one.
    if not chart.components:
        return _identity_dense(0)
    try:
        cart_chart = chart.cartesian
    except NoGlobalCartesianChartError:
        return _identity_dense(M.ndim)
    J = cxcapi.jac_pt_map(point, chart, cart_chart, usys=None)
    JT = J.T  # ty: ignore[unresolved-attribute]
    return DenseMetric(JT @ J)
