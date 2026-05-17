"""Vector Conversion."""

__all__ = ("change_basis",)

from jaxtyping import ArrayLike
from typing import Any, TypeVar

import jax
import jax.scipy.linalg
import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity, is_any_quantity

import coordinax.api.manifolds as cxmapi
import coordinax.api.representations as cxrapi
import coordinax.charts as cxc
import coordinax.manifolds as cxm
from .basis import (
    AbstractBasis,
    AbstractLinearBasis,
    CoordinateBasis,
    NoBasis,
    PhysicalBasis,
)
from .custom_types import CDict, OptUSys
from .geom import TangentGeometry
from .rep import Representation
from coordinax._src.metric.matrix import DenseMetric, DiagonalMetric
from coordinax.internal import QMatrix, UnitsMatrix

T = TypeVar("T", bound=u.Q)

_RAD = u.unit("rad")


def _drop_rad_unit(q: ArrayLike | u.AbstractQuantity) -> ArrayLike | u.AbstractQuantity:
    """Drop the "rad" unit from a quantity, otherwise return unchanged."""
    return BareQuantity(q.value, unit=q.unit / _RAD) if is_any_quantity(q) else q


def _add_rad_unit(q: ArrayLike | u.AbstractQuantity) -> ArrayLike | u.AbstractQuantity:
    """Add the "rad" unit to a quantity, otherwise return unchanged."""
    return BareQuantity(q.value, unit=q.unit * _RAD) if is_any_quantity(q) else q


def _qm_triangular_solve(E: QMatrix, b: QMatrix) -> QMatrix:
    """Solve upper-triangular system E @ x = b for x, respecting units.

    Uses the fact that E is upper-triangular (vielbein = L^T from Cholesky).
    Unit of x[i] is unit(b[i]) / unit(E[i,i]).

    Works on raw values: normalises E by its diagonal so E_norm is
    dimensionless (unit diagonal = 1), solves the normalised system, and
    reattaches output units.
    """
    n = E.unit.shape[0]
    x_units = UnitsMatrix(tuple(b.unit[i] / E.unit[i, i] for i in range(n)))
    # Scale b[i] → b_norm[i] = b.value[i] / E.value[i,i]  (dimensionless in
    # the sense that both numerator and denominator carry the same combined unit)
    # Since x[i] = b[i]/E[i,i] dimensionally, the raw solve gives the right
    # magnitude when we factor out the diagonal scaling from E.
    # D = diag(E.value[i,i]);  E_norm = D^{-1} E  (upper-triangular, unit diagonal)
    # D^{-1} b → b_norm[i] = b.value[i] / E.value[i,i]
    diag_vals = jnp.diagonal(E.value, axis1=-2, axis2=-1)
    b_norm = b.value / diag_vals  # shape (..., n), element-wise divide by diagonal
    E_norm = E.value / diag_vals[..., :, None]  # normalise each row by its diagonal
    x_vals = jax.scipy.linalg.solve_triangular(
        E_norm, b_norm[..., None], lower=False
    ).squeeze(-1)
    return QMatrix(x_vals, unit=x_units)


##############################################################################
# With a manifold


@plum.dispatch
def change_basis(
    v: CDict,
    chart: cxc.AbstractChart,
    M: cxm.AbstractManifold,
    from_basis: CoordinateBasis,
    to_basis: PhysicalBasis,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> CDict:
    r"""Change from coordinate basis to physical basis using a manifold.

    Retrieves the manifold's metric and applies the appropriate transformation.
    For diagonal metrics (e.g. `coordinax.manifolds.FlatMetric` in orthogonal
    charts) the fast scale-factor path is taken; for general metrics (e.g.
    `coordinax.manifolds.PullbackMetric`) the Cholesky vielbein $E = L^\top$ is
    used.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.representations as cxr

    Euclidean 3-D manifold in spherical coordinates (diagonal metric):

    >>> M3 = cxm.R3
    >>> v = {"r": u.Q(5, "m/s"), "theta": u.Q(1, "rad/s"), "phi": u.Q(2, "rad/s")}
    >>> at = {"r": u.Q(3, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0, "rad")}
    >>> cxr.change_basis(v, cxc.sph3d, M3, cxr.coord_basis, cxr.phys_basis, at=at)
    {'r': Q(5, 'm / s'), 'theta': Q(3, 'm / s'), 'phi': Q(2.87655323, 'm / s')}

    Embedded two-sphere manifold — non-diagonal
    :class:`~coordinax.manifolds.PullbackMetric`:

    >>> M = cxm.EmbeddedManifold(
    ...     intrinsic=cxm.S2, ambient=cxm.R3,
    ...     embed_map=cxm.TwoSphereIn3D(radius=u.Q(1.0, "km")),
    ... )
    >>> v = {"theta": u.Q(1.0, "rad/s"), "phi": u.Q(2.0, "rad/s")}
    >>> at = {"theta": u.Q(jnp.pi / 3, "rad"), "phi": u.Q(0.0, "rad")}
    >>> cxr.change_basis(v, cxc.sph2, M, cxr.coord_basis, cxr.phys_basis, at=at)
    {'theta': Q(..., 'km / s'), 'phi': Q(..., 'km / s')}

    """
    del from_basis, to_basis  # only used for dispatch

    at = chart.check_data(at, keys=True)
    keys = chart.components
    mm = cxmapi.metric_matrix(M, at, chart)
    if isinstance(mm, DiagonalMetric):
        h = jnp.sqrt(mm.diagonal)
        return {k: h[i] * v[k] for i, k in enumerate(keys)}
    # General case: Cholesky vielbein E = L^T, hat_v = E @ v
    assert isinstance(mm, DenseMetric)  # noqa: S101
    mat = mm.matrix
    if isinstance(mat, QMatrix):
        L_val = jnp.linalg.cholesky(mat.value)
        L_units = UnitsMatrix(mat.unit._units**0.5)
        L = QMatrix(L_val, unit=L_units)
    else:
        L_raw = jnp.linalg.cholesky(mat)
        n = mat.shape[-1]
        _dmls = u.unit("")
        L = QMatrix(
            L_raw,
            unit=UnitsMatrix(tuple(tuple(_dmls for _ in range(n)) for _ in range(n))),
        )
    E = jnp.transpose(L, axes=(-2, -1))  # E = L^T, upper-triangular vielbein
    v_vec = QMatrix.from_cdict(v, keys)
    hat_v_vec = jnp.matmul(E, v_vec)
    return cxc.cdict(hat_v_vec, keys)  # ty: ignore[invalid-return-type]


@plum.dispatch
def change_basis(
    v: CDict,
    chart: cxc.AbstractChart,
    M: cxm.AbstractManifold,
    from_basis: PhysicalBasis,
    to_basis: CoordinateBasis,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> CDict:
    r"""Change from physical basis to coordinate basis using a manifold.

    Retrieves the manifold's metric and applies the inverse transformation.  For
    diagonal metrics the fast scale-factor path is taken; for general metrics
    the Cholesky vielbein $E = L^\top$ is solved as a triangular system $v =
    E^{-1}\hat{v}$.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.representations as cxr

    Euclidean 3-D manifold in spherical coordinates (diagonal metric):

    >>> M3 = cxm.R3
    >>> v = {"r": u.Q(5, "m/s"), "theta": u.Q(3, "m/s"), "phi": u.Q(2.876553, "m/s")}
    >>> at = {"r": u.Q(3, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0.5, "rad")}
    >>> cxr.change_basis(v, cxc.sph3d, M3, cxr.phys_basis, cxr.coord_basis, at=at)
    {'r': Q(5, 'm / s'), 'theta': Q(1., 'rad / s'), 'phi': Q(1.99999..., 'rad / s')}

    Embedded two-sphere manifold — non-diagonal
    :class:`~coordinax.manifolds.PullbackMetric`:

    >>> M = cxm.EmbeddedManifold(
    ...     intrinsic=cxm.S2, ambient=cxm.R3,
    ...     embed_map=cxm.TwoSphereIn3D(radius=u.Q(1.0, "km")),
    ... )
    >>> v = {"theta": u.Q(1.0, "km/s"), "phi": u.Q(2.0, "km/s")}
    >>> at = {"theta": u.Q(jnp.pi / 3, "rad"), "phi": u.Q(0.0, "rad")}
    >>> cxr.change_basis(v, cxc.sph2, M, cxr.phys_basis, cxr.coord_basis, at=at)
    {'theta': Q(..., 'rad / s'), 'phi': Q(..., 'rad / s')}

    """
    del from_basis, to_basis  # only used for dispatch

    at = chart.check_data(at, keys=True)
    keys = chart.components
    mm = cxmapi.metric_matrix(M, at, chart)
    if isinstance(mm, DiagonalMetric):
        h = jnp.sqrt(mm.diagonal)
        return {k: v[k] / h[i] for i, k in enumerate(keys)}
    # General case: Cholesky vielbein E = L^T, v = E^{-1} hat_v (triangular solve)
    assert isinstance(mm, DenseMetric)  # noqa: S101
    mat = mm.matrix
    if isinstance(mat, QMatrix):
        L_val = jnp.linalg.cholesky(mat.value)
        L_units = UnitsMatrix(mat.unit._units**0.5)
        L = QMatrix(L_val, unit=L_units)
    else:
        L_raw = jnp.linalg.cholesky(mat)
        n = mat.shape[-1]
        _dmls = u.unit("")
        L = QMatrix(
            L_raw,
            unit=UnitsMatrix(tuple(tuple(_dmls for _ in range(n)) for _ in range(n))),
        )
    E = jnp.transpose(L, axes=(-2, -1))  # E = L^T, upper-triangular vielbein
    hat_v_vec = QMatrix.from_cdict(v, keys)
    v_vec = _qm_triangular_solve(E, hat_v_vec)
    return cxc.cdict(v_vec, keys)  # ty: ignore[invalid-return-type]


# ==============================================


@plum.dispatch
def change_basis(
    v: CDict,
    chart: cxc.AbstractChart,
    from_basis: PhysicalBasis,
    to_basis: CoordinateBasis,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> CDict:
    r"""Change from physical basis to coordinate basis using the chart's manifold.

    Falls back to ``chart.M`` when no explicit manifold is supplied.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> v = {"r": u.Q(5, "m/s"), "theta": u.Q(3, "m/s"), "phi": u.Q(2.876553, "m/s")}
    >>> at = {"r": u.Q(3, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0.5, "rad")}
    >>> cxr.change_basis(v, cxc.sph3d, cxr.phys_basis, cxr.coord_basis, at=at)
    {'r': Q(5, 'm / s'), 'theta': Q(1., 'rad / s'), 'phi': Q(1.99999984, 'rad / s')}

    """
    return cxrapi.change_basis(
        v, chart, chart.M, from_basis, to_basis, at=at, usys=usys
    )  # ty: ignore[invalid-return-type]


# ==============================================
# Special Cases


@plum.dispatch
def change_basis(
    v: CDict,
    chart: cxc.AbstractChart,
    from_basis: NoBasis | CoordinateBasis,
    to_basis: CoordinateBasis,
    /,
    **kw: Any,
) -> CDict:
    """Reinterpret unknown basis components as coordinate-basis components.

    This is an identity on component values: no numeric basis transform is
    possible from an unknown basis, so values are preserved and only the
    representation-level basis label changes.

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    The conversion is an identity map on values:

    >>> v = {"x": jnp.array(1.0), "y": jnp.array(-2.0)}
    >>> at = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
    >>> cxr.change_basis(v, cxc.cart2d, cxr.no_basis, cxr.coord_basis, at=at)
    {'x': Array(1., dtype=float64, ...), 'y': Array(-2., dtype=float64, ...)}

    "at" is accepted and ignored for this basis-only reinterpretation:

    >>> cxr.change_basis(v, cxc.cart2d, cxr.no_basis, cxr.coord_basis)
    {'x': Array(1., dtype=float64, ...), 'y': Array(-2., dtype=float64, ...)}

    """
    # TODO: check dimensions up to time powers
    del chart, from_basis, to_basis, kw
    return v


@plum.dispatch
def change_basis(
    v: CDict,
    chart: cxc.AbstractChart,
    from_basis: NoBasis | PhysicalBasis,
    to_basis: PhysicalBasis,
    /,
    **kw: Any,
) -> CDict:
    """Reinterpret unknown basis components as physical-basis components.

    This conversion is only well-defined when all components share the same
    physical dimension. No numeric transform is applied; values are preserved.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    Same-dimension components (even with different units) are accepted:

    >>> v = {"x": u.Q(1.0, "m / s"), "y": u.Q(2.0, "km / s")}
    >>> at = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
    >>> cxr.change_basis(v, cxc.cart2d, cxr.no_basis, cxr.phys_basis, at=at)
    {'x': Q(1., 'm / s'), 'y': Q(2., 'km / s')}

    Mixed dimensions are rejected:

    >>> v_bad = {"x": u.Q(1.0, "m / s"), "y": u.Q(2.0, "m")}
    >>> cxr.change_basis(v_bad, cxc.cart2d, cxr.no_basis, cxr.phys_basis)
    Traceback (most recent call last):
    ...
    ValueError: change_basis from NoBasis to PhysicalBasis requires all
    components to have the same dimension, got ...

    """
    chart.check_data(v, keys=True, values=False)
    dims = {u.dimension_of(value) for value in v.values()}
    if len(dims) > 1:
        msg = (
            "change_basis from NoBasis to PhysicalBasis requires all "
            f"components to have the same dimension, got {dims}"
        )
        raise ValueError(msg)

    del chart, from_basis, to_basis, kw
    return v


# ----------------------------------------------
# Cartesian


@plum.dispatch(precedence=1)  # ty: ignore[no-matching-overload]
def change_basis(
    v: CDict,
    chart: cxc.Cart0D | cxc.Cart1D | cxc.Cart2D | cxc.Cart3D | cxc.CartND,
    from_basis: CoordinateBasis | PhysicalBasis,
    to_basis: CoordinateBasis | PhysicalBasis,
    /,
    **kw: Any,
) -> CDict:
    r"""Change the basis used to interpret tangent components.

    In a Cartesian chart every scale factor equals one,

    $$h_x = h_y = h_z = 1,$$

    so the coordinate basis vectors are already unit vectors ($\hat{e}_i =
    \partial_i$) and the transformation matrix is the identity ($H = I$).  The
    coordinate basis **is** the physical basis, so this conversion is always the
    identity map and ``v`` is returned unchanged regardless of the direction of
    the conversion.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    Coordinate basis to physical basis in a 2-D Cartesian chart — identity:

    >>> v = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s")}
    >>> at = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
    >>> cxr.change_basis(v, cxc.cart2d, cxr.coord_basis, cxr.phys_basis, at=at)
    {'x': Q(3., 'm / s'), 'y': Q(4., 'm / s')}

    The reverse direction is equally a no-op:

    >>> cxr.change_basis(v, cxc.cart2d, cxr.phys_basis, cxr.coord_basis, at=at)
    {'x': Q(3., 'm / s'), 'y': Q(4., 'm / s')}

    Works for any Cartesian dimensionality; ``at`` is optional:

    >>> v3 = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")}
    >>> cxr.change_basis(v3, cxc.cart3d, cxr.coord_basis, cxr.phys_basis)
    {'x': Q(1., 'm / s'), 'y': Q(2., 'm / s'), 'z': Q(3., 'm / s')}

    """
    del chart, from_basis, to_basis, kw
    return v


@plum.dispatch(precedence=1)  #  ty: ignore[no-matching-overload]
def change_basis(
    v: CDict,
    chart: cxc.Cart0D | cxc.Cart1D | cxc.Cart2D | cxc.Cart3D | cxc.CartND,
    M: cxm.AbstractManifold,
    from_basis: AbstractLinearBasis,
    to_basis: AbstractLinearBasis,
    /,
    **kw: Any,
) -> CDict:
    r"""Change the basis used to interpret tangent components.

    In a Cartesian chart every scale factor equals one,

    $$h_x = h_y = h_z = 1,$$

    so the coordinate basis vectors are already unit vectors ($\hat{e}_i =
    \partial_i$) and the transformation matrix is the identity ($H = I$).  The
    coordinate basis **is** the physical basis, so this conversion is always the
    identity map and ``v`` is returned unchanged regardless of the direction of
    the conversion.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.representations as cxr

    Coordinate basis to physical basis in a 2-D Cartesian chart — identity:

    >>> v = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s")}
    >>> at = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
    >>> M2 = cxm.R2
    >>> cxr.change_basis(v, cxc.cart2d, M2, cxr.coord_basis, cxr.phys_basis, at=at)
    {'x': Q(3., 'm / s'), 'y': Q(4., 'm / s')}

    The reverse direction is equally a no-op:

    >>> cxr.change_basis(v, cxc.cart2d, M2, cxr.phys_basis, cxr.coord_basis, at=at)
    {'x': Q(3., 'm / s'), 'y': Q(4., 'm / s')}

    Works for any Cartesian dimensionality; ``at`` is optional:

    >>> v3 = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")}
    >>> cxr.change_basis(v3, cxc.cart3d, cxm.R3, cxr.coord_basis, cxr.phys_basis)
    {'x': Q(1., 'm / s'), 'y': Q(2., 'm / s'), 'z': Q(3., 'm / s')}

    """
    del chart, M, from_basis, to_basis, kw
    return v


# ----------------------------------------------
# Spherical


@plum.dispatch
def change_basis(
    v: CDict,
    chart: cxc.Spherical3D,
    from_basis: CoordinateBasis,
    to_basis: PhysicalBasis,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> CDict:
    r"""Change from coordinate basis to physical basis in a 3-D spherical chart.

    In spherical coordinates $(r, \theta, \phi)$ the scale factors are

    $$h_r = 1, \quad h_\theta = r, \quad h_\phi = r \sin\theta,$$

    so the transformation matrix is

    $$
    H = \begin{pmatrix}
        1 & 0 & 0 \\
        0 & r & 0 \\
        0 & 0 & r\sin\theta
    \end{pmatrix}.
    $$

    Given coordinate-basis components $(v^r, v^\theta, v^\phi)$, the
    physical-basis components are

    $$
    \hat{v} = H v
        \implies
        \hat{v}^r = v^r, \quad
        \hat{v}^\theta = r\, v^\theta, \quad
        \hat{v}^\phi = r\sin\theta\, v^\phi.
    $$

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> v = {"r": u.Q(5, "m/s"), "theta": u.Q(1, "rad/s"), "phi": u.Q(2, "rad/s")}
    >>> at = {"r": u.Q(3, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0, "rad")}
    >>> cxr.change_basis(v, cxc.sph3d, cxr.coord_basis, cxr.phys_basis, at=at)
    {'r': Q(5, 'm / s'), 'theta': Q(3, 'm / s'), 'phi': Q(2.87655323, 'm / s')}

    >>> v = {"r": 5, "theta": 1, "phi": 2}  # unitless
    >>> at = {"r": 3, "theta": 0.5, "phi": 0}  # unitless
    >>> cxr.change_basis(v, cxc.sph3d, cxr.coord_basis, cxr.phys_basis, at=at)
    {'r': 5, 'theta': 3, 'phi': Array(2.87655323, dtype=float64, ...)}

    """
    del chart, usys
    r = at["r"]
    return {
        "r": v["r"],
        "theta": r * _drop_rad_unit(v["theta"]),
        "phi": r * jnp.sin(at["theta"]) * _drop_rad_unit(v["phi"]),
    }


@plum.dispatch
def change_basis(
    v: CDict,
    chart: cxc.Spherical3D,
    M: cxm.EuclideanManifold,
    from_basis: CoordinateBasis,
    to_basis: PhysicalBasis,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> CDict:
    r"""Change from coordinate basis to physical basis in a 3-D spherical chart.

    Delegates to the chart-specific implementation for
    `coordinax.manifolds.EuclideanManifold`.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.representations as cxr

    >>> v = {"r": u.Q(5, "m/s"), "theta": u.Q(1, "rad/s"), "phi": u.Q(2, "rad/s")}
    >>> at = {"r": u.Q(3, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0, "rad")}
    >>> cxr.change_basis(v, cxc.sph3d, cxm.R3, cxr.coord_basis, cxr.phys_basis, at=at)
    {'r': Q(5, 'm / s'), 'theta': Q(3, 'm / s'), 'phi': Q(2.87655323, 'm / s')}

    >>> v = {"r": 5, "theta": 1, "phi": 2}  # unitless
    >>> at = {"r": 3, "theta": 0.5, "phi": 0}  # unitless
    >>> cxr.change_basis(v, cxc.sph3d, cxm.R3, cxr.coord_basis, cxr.phys_basis, at=at)
    {'r': 5, 'theta': 3, 'phi': Array(2.87655323, dtype=float64, ...)}

    """
    del M
    return cxrapi.change_basis(v, chart, from_basis, to_basis, at=at, usys=usys)  # ty: ignore[invalid-return-type]


# ------------


@plum.dispatch
def change_basis(
    v: CDict,
    chart: cxc.Spherical3D,
    from_basis: PhysicalBasis,
    to_basis: CoordinateBasis,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> CDict:
    r"""Change from physical basis to coordinate basis in a 3-D spherical chart.

    In spherical coordinates $(r, \theta, \phi)$ the inverse transformation
    matrix is

    $$
    H^{-1} = \begin{pmatrix}
        1 & 0 & 0 \\
        0 & 1/r & 0 \\
        0 & 0 & 1/(r\sin\theta)
    \end{pmatrix}.
    $$

    Given physical-basis components $(\hat{v}^r, \hat{v}^\theta, \hat{v}^\phi)$,
    the coordinate-basis components are

    $$
    v = H^{-1} \hat{v}
        \implies
        v^r = \hat{v}^r, \quad
        v^\theta = \hat{v}^\theta / r, \quad
        v^\phi = \hat{v}^\phi / (r\sin\theta).
    $$

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> v = {"r": u.Q(5, "m/s"), "theta": u.Q(3, "m/s"), "phi": u.Q(2.876553, "m/s")}
    >>> at = {"r": u.Q(3, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0.5, "rad")}
    >>> cxr.change_basis(v, cxc.sph3d, cxr.phys_basis, cxr.coord_basis, at=at)
    {'r': Q(5, 'm / s'), 'theta': Q(1., 'rad / s'), 'phi': Q(1.99999984, 'rad / s')}

    >>> v = {"r": 5, "theta": 3, "phi": 2.876553}  # unitless
    >>> at = {"r": 3, "theta": 0.5, "phi": 0.5}  # unitless
    >>> cxr.change_basis(v, cxc.sph3d, cxr.phys_basis, cxr.coord_basis, at=at)
    {'r': 5, 'theta': 1.0, 'phi': Array(1.99999984, dtype=float64, ...)}

    """
    del chart, usys
    r = at["r"]
    return {
        "r": v["r"],
        "theta": _add_rad_unit(v["theta"]) / r,
        "phi": _add_rad_unit(v["phi"]) / (r * jnp.sin(at["theta"])),
    }


@plum.dispatch
def change_basis(
    v: CDict,
    chart: cxc.Spherical3D,
    M: cxm.EuclideanManifold,
    from_basis: PhysicalBasis,
    to_basis: CoordinateBasis,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> CDict:
    r"""Change from physical basis to coordinate basis in a 3-D spherical chart.

    Delegates to the chart-specific implementation for
    :class:`~coordinax.manifolds.EuclideanManifold`.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.representations as cxr

    >>> v = {"r": u.Q(5, "m/s"), "theta": u.Q(3, "m/s"), "phi": u.Q(2.876553, "m/s")}
    >>> at = {"r": u.Q(3, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0.5, "rad")}
    >>> cxr.change_basis(v, cxc.sph3d, cxm.R3, cxr.phys_basis, cxr.coord_basis, at=at)
    {'r': Q(5, 'm / s'), 'theta': Q(1., 'rad / s'), 'phi': Q(1.99999984, 'rad / s')}

    >>> v = {"r": 5, "theta": 3, "phi": 2.876553}  # unitless
    >>> at = {"r": 3, "theta": 0.5, "phi": 0.5}  # unitless
    >>> cxr.change_basis(v, cxc.sph3d, cxm.R3, cxr.phys_basis, cxr.coord_basis, at=at)
    {'r': 5, 'theta': 1.0, 'phi': Array(1.99999984, dtype=float64, ...)}

    """
    del M
    return cxrapi.change_basis(v, chart, from_basis, to_basis, at=at, usys=usys)  # ty: ignore[invalid-return-type]


#####################################################################


@plum.dispatch.multi(
    (CDict, cxc.AbstractChart, Representation, Representation),
    (CDict, cxc.AbstractChart, AbstractBasis, Representation),
    (CDict, cxc.AbstractChart, Representation, AbstractBasis),
)
def change_basis(
    v: CDict,
    chart: cxc.AbstractChart,
    from_rep: Representation,
    to_rep: Representation,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> CDict:
    """Change basis using source and/or target :class:`Representation` objects.

    This is a convenience overload: the caller may pass full
    :class:`Representation` objects for ``from_rep``/``to_rep`` instead of bare
    :class:`AbstractBasis` instances.  The basis is extracted from each argument
    and the appropriate :func:`change_basis` overload is called.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    Coordinate-basis displacement to physical-basis displacement in a
    spherical chart, passing full :class:`Representation` objects:

    >>> v = {"r": u.Q(5, "m/s"), "theta": u.Q(1, "rad/s"), "phi": u.Q(2, "rad/s")}
    >>> at = {"r": u.Q(3, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0, "rad")}
    >>> cxr.change_basis(v, cxc.sph3d, cxr.coord_disp, cxr.phys_disp, at=at)
    {'r': Q(5, 'm / s'), 'theta': Q(3, 'm / s'), 'phi': Q(2.87655323, 'm / s')}

    >>> v = {"r": 5, "theta": 1, "phi": 2}  # unitless
    >>> at = {"r": 3, "theta": 0.5, "phi": 0}  # unitless
    >>> cxr.change_basis(v, cxc.sph3d, cxr.coord_disp, cxr.phys_disp, at=at)
    {'r': 5, 'theta': 3, 'phi': Array(2.87655323, dtype=float64, ...)}

    """
    if isinstance(from_rep, Representation) and not isinstance(
        from_rep.geom_kind, TangentGeometry
    ):
        msg = "change_basis with Representation requires TangentGeometry for from_rep"
        raise TypeError(msg)
    if isinstance(to_rep, Representation) and not isinstance(
        to_rep.geom_kind, TangentGeometry
    ):
        msg = "change_basis with Representation requires TangentGeometry for to_rep"
        raise TypeError(msg)

    # This multi-dispatch handles the case where the caller provides one or both
    # of the representations instead of the bases. We just extract the bases and
    # call the main dispatch.
    from_basis = from_rep if isinstance(from_rep, AbstractBasis) else from_rep.basis
    to_basis = to_rep if isinstance(to_rep, AbstractBasis) else to_rep.basis
    # Re-dispatch to the main implementation.
    return cxrapi.change_basis(v, chart, from_basis, to_basis, at=at, usys=usys)  # ty: ignore[invalid-return-type]
