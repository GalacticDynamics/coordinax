"""Shared base for pure linear (matrix) transforms.

`Rotate`, `Scale`, `Shear`, and `Reflect` all act as ``x -> M x`` in the
canonical Cartesian chart. This module owns that shared machinery — matrix
validation, the point-action kernel (chart -> cartesian -> einsum -> back), the
Array / Quantity fast paths, and the Cartesian-product factorwise dispatch — so
each operator only supplies its matrix via `_raw_matrix`.
"""

__all__ = ("AbstractLinearTransform",)

from abc import abstractmethod

from jaxtyping import Array, ArrayLike
from typing import Any, cast

import equinox as eqx
import plum

import quaxed.numpy as jnp
from unxt import AbstractQuantity as AbcQ

import coordinax.api.transforms as cxfmapi
import coordinax.charts as cxc
import coordinax.representations as cxr
from .base import AbstractTransform, materialize_transform
from .custom_types import CDict, HasShape, OptUSys
from coordinax.internal import pack_uniform_unit


class AbstractLinearTransform(AbstractTransform):
    r"""Base for pure Cartesian linear maps :math:`x \mapsto M x`.

    A subclass provides its matrix via the `_raw_matrix` property (which may be
    a callable of ``tau`` for a time-dependent map); this base owns the matrix
    validation and every point-geometry ``act`` path.
    """

    @property
    @abstractmethod
    def _raw_matrix(self) -> Any:
        """The matrix parameter (an array, or a callable of ``tau``)."""
        raise NotImplementedError  # pragma: no cover

    def _validate_square(self, matrix: HasShape, /) -> Array:
        """Check the matrix is square (N x N)."""
        shape = matrix.shape
        return eqx.error_if(
            matrix,
            len(shape) != 2 or shape[0] != shape[1],
            f"{type(self).__name__} requires a square matrix; got shape {shape!r}.",
        )

    def _validate_shape_match(
        self, matrix: Array, cart: cxc.AbstractChart[Any, Any, Any], /
    ) -> Array:
        """Check the matrix dimension matches the Cartesian chart dimension."""
        n = matrix.shape[0]
        return eqx.error_if(
            matrix,
            cart.ndim != n or len(cart.components) != n,
            f"{type(self).__name__}: matrix dimension {n} does not match the "
            f"canonical Cartesian chart {type(cart).__name__} (ndim={cart.ndim!r}).",
        )

    def _matrix(
        self, cart: cxc.AbstractChart[Any, Any, Any], tau: Any = None, /
    ) -> Array:
        """Return the validated matrix for ``cart``, materialized at ``tau``."""
        op_eval = materialize_transform(self, tau)
        matrix = op_eval._raw_matrix
        matrix = eqx.error_if(
            matrix, callable(matrix), "need to call `materialize_transform`."
        )
        matrix = self._validate_square(matrix)
        return self._validate_shape_match(matrix, cart)


# ============================================================================
# act — point geometry (shared by every linear transform)


@plum.dispatch
def act(
    op: AbstractLinearTransform,
    tau: Any,
    x: ArrayLike,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> Array:
    """Apply a linear transform to an Array(like) object."""
    del kw  # Does not require an anchoring base-point.

    x_arr = jnp.asarray(x)
    chart = cxc.guess_chart(x_arr)  # ty: ignore[invalid-assignment]
    if chart != chart.cartesian:
        msg = (
            f"act for {type(op).__name__} with ArrayLike x requires a Cartesian chart."
        )
        raise ValueError(msg)
    if rep != cxr.point:
        msg = (
            f"act for {type(op).__name__} with ArrayLike x requires a "
            "point representation."
        )
        raise TypeError(msg)

    matrix = op._matrix(chart, tau)
    return jnp.einsum("ij,...j->...i", matrix, x_arr)


@plum.dispatch
def act(
    op: AbstractLinearTransform,
    tau: Any,
    x: AbcQ,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> AbcQ:
    """Apply a linear transform to a PointGeometry-roled Quantity."""
    del rep, kw

    cart = chart.cartesian
    if chart != cart:
        msg = (
            f"act({type(op).__name__}, ..., Quantity) requires Cartesian "
            f"components. chart {type(chart).__name__} is not its cartesian_chart."
        )
        raise ValueError(msg)

    matrix = op._matrix(cart, tau)
    return jnp.einsum("ij,...j->...i", matrix, x)  # ty: ignore[invalid-return-type]


@plum.dispatch
def act(
    op: AbstractLinearTransform,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    *,
    usys: OptUSys = None,
    **kw: Any,
) -> CDict:
    """Redispatch a CDict to the geometry-specific implementation."""
    out = cxfmapi.act(op, tau, x, chart, rep.geom_kind, rep, usys=usys, **kw)
    return cast("CDict", out)


@plum.dispatch
def act(
    op: AbstractLinearTransform,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    geom: cxr.PointGeometry,
    rep: cxr.Representation,
    /,
    *,
    usys: OptUSys = None,
    **kw: Any,
) -> CDict:
    """Apply a linear map to a Point-valued coordinate dictionary.

    The point is mapped by converting to the chart's canonical Cartesian chart,
    applying the matrix in Cartesian components, then converting back. Units are
    handled by packing Cartesian components into a common unit before the map
    and restoring it afterward.
    """
    del geom, rep, kw  # Does not require an anchoring base-point.

    cart = chart.cartesian
    comps_cart = cart.components
    matrix = op._matrix(cart, tau)

    p_cart = cxc.pt_map(x, chart, cart, usys=usys)

    v, unit = pack_uniform_unit(p_cart, keys=comps_cart)  # ty: ignore[no-matching-overload]
    v_out = jnp.einsum("ij,...j->...i", matrix, v)
    p_cart_out = cxc.cdict(v_out, unit, comps_cart)

    out = cxc.pt_map(p_cart_out, cart, chart, usys=usys)
    return cast("CDict", out)


@plum.dispatch
def act(
    op: AbstractLinearTransform,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractCartesianProductChart,
    geom: cxr.PointGeometry,
    rep: cxr.Representation,
    /,
    *,
    usys: OptUSys = None,
    **kw: Any,
) -> CDict:
    """Apply a linear map factorwise on Cartesian-product charts."""
    n = op._validate_square(materialize_transform(op, tau)._raw_matrix).shape[-1]

    n_factors = len(chart.factors)
    parts = chart.split_components(x)
    ats = {
        k: chart.split_components(v) if v is not None else [None] * n_factors
        for k, v in kw.items()
        if k.startswith("at")
    }

    def _maybe(
        factor_chart: cxc.AbstractChart[Any, Any, Any],
        part: CDict,
        /,
        **ats: CDict | None,
    ) -> CDict:
        cart = factor_chart.cartesian
        if cart.ndim != n or len(cart.components) != n:
            return part

        out = cxfmapi.act(op, tau, part, factor_chart, geom, rep, usys=usys, **ats)
        return cast("CDict", out)

    mapped_parts = tuple(
        _maybe(f, p, **{k: splits[i] for k, splits in ats.items()})
        for i, (f, p) in enumerate(zip(chart.factors, parts, strict=True))
    )
    return chart.merge_components(mapped_parts)
