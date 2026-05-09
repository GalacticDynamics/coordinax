"""Tangent map (Jacobian pushforward) between coordinate charts."""

__all__ = ("tangent_map",)

from jaxtyping import Array
from typing import Any

import jax.numpy as jnp
import plum

import quaxed.numpy as qnp
import unxt as u

import coordinax.api.representations as cxrapi
import coordinax.charts as cxc
from .basis import CoordinateBasis, PhysicalBasis, coord_basis
from .custom_types import CDict, OptUSys
from .geom import TangentGeometry
from .rep import Representation
from coordinax.internal import QuantityMatrix, pack_nonuniform_unit

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_tangent_geom(geom: object, label: str) -> None:
    if not isinstance(geom, TangentGeometry):
        raise TypeError(
            f"tangent_map requires TangentGeometry for {label}, got {geom!r}"
        )


def _check_linear_basis(rep: Representation, label: str) -> None:
    if not isinstance(rep.basis, CoordinateBasis | PhysicalBasis):
        raise TypeError(
            "tangent_map requires CoordinateBasis or PhysicalBasis for "
            f"{label}, got {rep.basis!r}"
        )


# ---------------------------------------------------------------------------
# Shared helper: apply a QuantityMatrix Jacobian to a tangent vector CDict
# ---------------------------------------------------------------------------


def _apply_jac(
    J: Array | QuantityMatrix,
    from_components: tuple[str, ...],
    to_components: tuple[str, ...],
    v: CDict,
) -> CDict:
    """Apply a 2-D QuantityMatrix Jacobian to a tangent CDict.

    If the components of ``v`` are plain arrays, the output is a plain-array
    CDict (using ``J.value @ v_arr``).  If any component of ``v`` is a
    {class}`~unxt.AbstractQuantity`, ``v`` is packed into a 1-D
    {class}`~coordinax.internal.QuantityMatrix` and the result is computed
    via ``qnp.matmul(J, v_qm)``, which handles per-element unit conversion.

    Parameters
    ----------
    J
        QuantityMatrix of shape ``(n_out, n_in)`` returned by ``jac_pt_map``.
    from_components
        Ordered component names for the input chart (columns of J).
    to_components
        Ordered component names for the output chart (rows of J).
    v
        Tangent vector components.  Values may be plain JAX arrays or
        {class}`~unxt.AbstractQuantity` objects.

    Returns
    -------
    CDict
        Tangent vector components in the output chart.

    """
    if isinstance(v[from_components[0]], u.AbstractQuantity):
        v_arr, v_units = pack_nonuniform_unit(v, keys=from_components)
        v_qm = QuantityMatrix(v_arr, unit=v_units)
        w = qnp.matmul(J, v_qm)  # (n_out,) QuantityMatrix
        return {key: u.Q(w.value[i], w.unit[i]) for i, key in enumerate(to_components)}

    v_arr = jnp.stack([jnp.asarray(v[k]) for k in from_components])
    # When J is a QuantityMatrix, use J.value to avoid the Quax fallback path
    # that returns a QuantityMatrix with J's own 2D unit structure (wrong).
    # Plain-array velocity is dimensionless, so numeric-only application is correct.
    j_arr = J.value if isinstance(J, QuantityMatrix) else J
    result = j_arr @ v_arr
    return {key: result[i] for i, key in enumerate(to_components)}


# ---------------------------------------------------------------------------
# Same-Representation dispatch
# ---------------------------------------------------------------------------


@plum.dispatch
def tangent_map(
    v: Any,
    from_chart: cxc.AbstractChart,
    basis: CoordinateBasis,
    to_chart: cxc.AbstractChart,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
) -> CDict:
    r"""Push a tangent vector forward from one chart to another.

    Applies the Jacobian of the chart transition map to the tangent vector
    components ``v``, evaluated at the base point ``at``.

    Examples
    --------
    Convert a tangent vector from Cartesian to polar 2D at the point (1, 0):

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> v = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
    >>> at = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
    >>> cxr.tangent_map(v, cxc.cart2d, cxr.coord_disp, cxc.polar2d, at=at)
    {'r': Array(1., dtype=float64), 'theta': Array(0., dtype=float64)}

    """
    # Same-chart optimization: identity transform
    if from_chart is to_chart:
        return v

    J = cxc.jac_pt_map(at, from_chart, to_chart, usys=usys)
    return _apply_jac(J, from_chart.components, to_chart.components, v)


@plum.dispatch
def tangent_map(
    v: CDict,
    from_chart: cxc.AbstractChart,
    basis: PhysicalBasis,
    to_chart: cxc.AbstractChart,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
) -> CDict:
    r"""Push a tangent vector forward in physical-basis components.

    This dispatch applies the tangent-map pushforward while preserving the
    physical-basis convention by composing three steps:

    1. convert source components from physical basis to coordinate basis,
    2. apply the chart Jacobian pushforward,
    3. convert target components back to physical basis.

    Examples
    --------
    Convert a physical-basis tangent vector from Cartesian to spherical 3D:

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> v = {"x": u.Q(1, "m/s"), "y": u.Q(0, "m/s"), "z": u.Q(0, "m/s")}
    >>> at = {"x": u.Q(1, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")}
    >>> cxr.tangent_map(v, cxc.cart3d, cxr.phys_basis, cxc.sph3d, at=at)
    {'r': Q(1., 'm / s'), 'theta': Q(-0., 'm / s'), 'phi': Q(0., 'm / s')}

    The same call can be made using a physical representation:

    >>> v = {"x": 1, "y": 0, "z": 0}
    >>> at = {"x": 1, "y": 0, "z": 0}
    >>> usys = u.unitsystems.si
    >>> cxr.tangent_map(v, cxc.cart3d, cxr.phys_disp, cxc.sph3d, at=at, usys=usys)
    {'r': Array(1., dtype=float64), 'theta': Array(0., dtype=float64),
     'phi': Array(0., dtype=float64)}

    """
    # Same-chart optimization: identity transform
    if from_chart is to_chart:
        return v

    # TODO: direct routes
    # Compute physical-basis transport by composing:
    #   physical -> coordinate -> Jacobian pushforward -> physical

    # Basis Change: physical to coord
    v_coord = cxrapi.change_basis(v, from_chart, basis, coord_basis, at=at, usys=usys)
    # Chart Jacobian pushforward in coordinate basis
    v_coord_to = cxrapi.tangent_map(
        v_coord, from_chart, coord_basis, to_chart, at=at, usys=usys
    )
    at_to = cxc.pt_map(at, from_chart, to_chart, usys=usys)
    # Basis Change: coord to physical
    v_coord: CDict = cxrapi.change_basis(  # ty: ignore[invalid-assignment]
        v_coord_to, to_chart, coord_basis, basis, at=at_to, usys=usys
    )
    return v_coord


@plum.dispatch
def tangent_map(
    v: Any,
    from_chart: cxc.AbstractChart,
    from_rep: Representation,
    to_chart: cxc.AbstractChart,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
) -> CDict:
    r"""Push a tangent vector forward from one chart to another.

    Applies the Jacobian of the chart transition map to the tangent vector
    components ``v``, evaluated at the base point ``at``.

    Examples
    --------
    Convert a tangent vector from Cartesian to polar 2D at the point (1, 0):

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> v = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
    >>> at = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
    >>> cxr.tangent_map(v, cxc.cart2d, cxr.coord_disp, cxc.polar2d, at=at)
    {'r': Array(1., dtype=float64), 'theta': Array(0., dtype=float64)}

    """
    _check_tangent_geom(from_rep.geom_kind, "from_rep")

    return cxrapi.tangent_map(v, from_chart, from_rep.basis, to_chart, at=at, usys=usys)  # ty: ignore[invalid-return-type]


# ---------------------------------------------------------------------------
# Cross-Representation dispatch
# ---------------------------------------------------------------------------


@plum.dispatch
def tangent_map(
    v: Any,
    from_chart: cxc.AbstractChart,
    from_rep: Representation,
    to_chart: cxc.AbstractChart,
    to_rep: Representation,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
) -> CDict:
    r"""Push a tangent vector forward from one chart to another.

    Applies the Jacobian of the chart transition map to the tangent vector
    components ``v``, evaluated at the base point ``at``.

    Examples
    --------
    Convert a tangent vector from Cartesian to polar 2D at the point (1, 0):

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> v = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
    >>> at = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
    >>> cxr.tangent_map(v, cxc.cart2d, cxr.coord_disp,
    ...                 cxc.polar2d, cxr.coord_disp, at=at)
    {'r': Array(1., dtype=float64), 'theta': Array(0., dtype=float64)}

    """
    v = cxrapi.tangent_map(v, from_chart, from_rep, to_chart, at=at, usys=usys)
    v = cxrapi.change_basis(v, to_chart, from_rep.basis, to_rep.basis, at=at, usys=usys)
    return v  # noqa: RET504  # ty: ignore[invalid-return-type]
