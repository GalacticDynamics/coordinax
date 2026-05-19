"""Register ``metric_matrix`` and ``metric_representation`` dispatch rules.

Covers :class:`~coordinax.manifolds.EuclideanManifold` paired with every
chart in its atlas.  The rules follow a two-tier scheme:

* **Cartesian charts** (``Cart0D``, ``Cart1D``, ``Cart2D``, ``Cart3D``,
  ``CartND``) and **orthogonal curvilinear charts** (``Radial1D``,
  ``Polar2D``, ``Cylindrical3D``, ``Spherical3D``, ``MathSpherical3D``,
  ``LonLatSpherical3D``) have explicit analytic diagonal metrics and return
  a :class:`~coordinax._src.metric.matrix.DiagonalMetric`.
* **All other charts** compute the Jacobian pullback ``g = J^T J`` directly
  and return the result as a :class:`~coordinax._src.metric.matrix.DenseMetric`.

"""

__all__: tuple[str, ...] = ()

from typing import Any, cast

import jax.numpy as jnp
import plum

import unxt as u

import coordinax.api.charts as cxcapi
from .manifold import EuclideanManifold
from coordinax._src.base import AbstractChart  # type: ignore[type-arg]
from coordinax._src.charts.d1 import Cart1D, Radial1D
from coordinax._src.charts.d2 import Cart2D, Polar2D
from coordinax._src.charts.d3 import (
    Cart3D,
    Cylindrical3D,
    LonLatSpherical3D,
    MathSpherical3D,
    Spherical3D,
)
from coordinax._src.charts.dn import CartND
from coordinax._src.exceptions import NoGlobalCartesianChartError
from coordinax._src.metric.matrix import DenseMetric, DiagonalMetric
from coordinax.internal import QMatrix, UnitsMatrix

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


def _angle_unit(q: Any, /) -> u.AbstractUnit:
    """Return the unit of an angular coordinate, or dimensionless if plain array."""
    if isinstance(q, u.AbstractQuantity):
        return cast("u.AbstractUnit", q.unit)
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
    >>> from coordinax.api.manifolds import metric_representation
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
    | LonLatSpherical3D,
    /,
) -> type[DiagonalMetric]:
    """Euclidean manifold in a Cartesian or orthogonal curvilinear chart.

    Returns :class:`DiagonalMetric`.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinax.api.manifolds import metric_representation
    >>> from coordinax._src.metric.matrix import DiagonalMetric

    >>> metric_representation(cxm.R3, cxc.cart3d)
    <class 'coordinax._src.metric.matrix.DiagonalMetric'>

    >>> metric_representation(cxm.R2, cxc.polar2d)
    <class 'coordinax._src.metric.matrix.DiagonalMetric'>

    >>> metric_representation(cxm.R3, cxc.sph3d)
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
    >>> from coordinax.api.manifolds import metric_matrix
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
    >>> from coordinax.api.manifolds import metric_matrix

    >>> at = {"q": jnp.array([1.0, 2.0, 3.0])}
    >>> metric_matrix(cxm.R3, at, cxc.cartnd).diagonal
    Array([1., 1., 1.], dtype=float64)

    """
    del M, chart
    n = jnp.asarray(point["q"]).shape[0]
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
    >>> from coordinax.api.manifolds import metric_matrix
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
    QMatrix([1.], '(,)')

    """
    del M, point, chart
    dmls = u.unit("")
    return DiagonalMetric(QMatrix(jnp.ones(1), unit=UnitsMatrix((dmls,))))


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
    >>> from coordinax.api.manifolds import metric_matrix
    >>> from coordinax._src.metric.matrix import DiagonalMetric

    Dimensionless ``r``:

    >>> at = {"r": jnp.array(3.0), "theta": jnp.array(0.5)}
    >>> g = metric_matrix(cxm.R2, at, cxc.polar2d)
    >>> g.diagonal
    QMatrix([1., 9.], '(, )')

    Length-valued ``r`` and angle-valued ``theta``:

    >>> at = {"r": u.Q(3.0, "m"), "theta": u.Angle(0.5, "rad")}
    >>> g = metric_matrix(cxm.R2, at, cxc.polar2d)
    >>> g.diagonal
    QMatrix([1., 9.], '(, m2 / rad2)')

    """
    del M, chart
    r_val, r_unit = _val_unit(point["r"])
    theta_unit = _angle_unit(point["theta"])
    diag = jnp.stack([jnp.asarray(1.0), r_val**2])
    units = UnitsMatrix((u.unit(""), r_unit**2 / theta_unit**2))
    return DiagonalMetric(QMatrix(diag, unit=units))


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
    >>> from coordinax.api.manifolds import metric_matrix
    >>> from coordinax._src.metric.matrix import DiagonalMetric

    >>> at = {"rho": u.Q(3.0, "m"), "phi": u.Angle(0.0, "rad"), "z": u.Q(1.0, "m")}
    >>> g = metric_matrix(cxm.R3, at, cxc.cyl3d)
    >>> g.diagonal
    QMatrix([1., 9., 1.], '(, m2 / rad2, )')

    """
    del M, chart
    rho_val, rho_unit = _val_unit(point["rho"])
    phi_unit = _angle_unit(point["phi"])
    dmls = u.unit("")
    diag = jnp.stack([jnp.asarray(1.0), rho_val**2, jnp.asarray(1.0)])
    units = UnitsMatrix((dmls, rho_unit**2 / phi_unit**2, dmls))
    return DiagonalMetric(QMatrix(diag, unit=units))


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
    >>> from coordinax.api.manifolds import metric_matrix
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
    QMatrix([1., 4., 4.], '(, m2 / rad2, m2 / rad2)')

    """
    del M, chart
    r_val, r_unit = _val_unit(point["r"])
    theta_val = _angle_rad(point["theta"])
    theta_unit = _angle_unit(point["theta"])
    phi_unit = _angle_unit(point["phi"])
    r2 = r_val**2
    r2_unit = r_unit**2
    diag = jnp.stack([jnp.asarray(1.0), r2, r2 * jnp.sin(theta_val) ** 2])
    units = UnitsMatrix((u.unit(""), r2_unit / theta_unit**2, r2_unit / phi_unit**2))
    return DiagonalMetric(QMatrix(diag, unit=units))


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
    >>> from coordinax.api.manifolds import metric_matrix
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
    QMatrix([1., 4., 4.], '(, m2 / rad2, m2 / rad2)')

    """
    del M, chart
    r_val, r_unit = _val_unit(point["r"])
    phi_val = _angle_rad(point["phi"])  # polar / colatitude angle
    theta_unit = _angle_unit(point["theta"])
    phi_unit = _angle_unit(point["phi"])
    r2 = r_val**2
    r2_unit = r_unit**2
    diag = jnp.stack([jnp.asarray(1.0), r2 * jnp.sin(phi_val) ** 2, r2])
    units = UnitsMatrix((u.unit(""), r2_unit / theta_unit**2, r2_unit / phi_unit**2))
    return DiagonalMetric(QMatrix(diag, unit=units))


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
    >>> from coordinax.api.manifolds import metric_matrix
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
    QMatrix([4., 4., 1.], '(m2 / rad2, m2 / rad2, )')

    """
    del M, chart
    d_val, d_unit = _val_unit(point["distance"])
    lat_val = _angle_rad(point["lat"])
    lon_unit = _angle_unit(point["lon"])
    lat_unit = _angle_unit(point["lat"])
    d2 = d_val**2
    d2_unit = d_unit**2
    diag = jnp.stack([d2 * jnp.cos(lat_val) ** 2, d2, jnp.asarray(1.0)])
    units = UnitsMatrix((d2_unit / lon_unit**2, d2_unit / lat_unit**2, u.unit("")))
    return DiagonalMetric(QMatrix(diag, unit=units))


# =====================================================================
# metric_matrix — General fallback (Jacobian pullback)
# =====================================================================


@plum.dispatch
def metric_matrix(
    M: EuclideanManifold, point: dict, chart: AbstractChart, /
) -> DenseMetric:
    """Euclidean metric in a general chart via Jacobian pullback ``g = J^T J``.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinax.api.manifolds import metric_matrix
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
    try:
        cart_chart = chart.cartesian
    except NoGlobalCartesianChartError:
        n = M.ndim
        unit_tup = tuple(tuple(u.unit("") for _ in range(n)) for _ in range(n))
        return DenseMetric(QMatrix(jnp.eye(n), unit=UnitsMatrix(unit_tup)))
    J = cxcapi.jac_pt_map(point, chart, cart_chart, usys=None)
    JT = J.T  # ty: ignore[unresolved-attribute]
    return DenseMetric(JT @ J)
