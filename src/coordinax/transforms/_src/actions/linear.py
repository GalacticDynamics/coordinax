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

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinaxs.api.transforms as cxfmapi
from .base import AbstractTransform
from .custom_types import CDict, HasShape, OptUSys
from .utils import is_flat_chart, require_matching_keys
from coordinax.internal import pack_uniform_unit


def _matmul_cdict(matrix: Array, d: CDict, comps: tuple[str, ...], /) -> CDict:
    """Apply ``matrix`` to a Cartesian cdict ``d``, packed into a shared unit."""
    v, unit = pack_uniform_unit(d, keys=comps)
    return cast("CDict", cxc.cdict(jnp.einsum("ij,...j->...i", matrix, v), unit, comps))


class AbstractLinearTransform(AbstractTransform):
    r"""Base for pure Cartesian linear maps :math:`x \mapsto M x`.

    A subclass provides its (constant) matrix via the `_raw_matrix` property;
    this base owns the matrix validation and every point-geometry ``act`` path.
    A time-dependent linear map is a `~coordinax.transforms.TimeDep` family of
    these operators.
    """

    @property
    @abstractmethod
    def _raw_matrix(self) -> Any:
        """The constant matrix parameter, as the subclass stores it."""
        raise NotImplementedError  # pragma: no cover

    @property
    def matrix(self) -> Array:
        r"""The matrix $M$ this transform applies, as $x \mapsto M x$.

        The one spelling shared by every subclass, which otherwise keep $M$
        under their own letter (``R``, ``H``, ``S``) or, for
        `~coordinax.transforms.LorentzBoost`, derive it from a stored velocity.

        Validated square, but not against a chart -- a chart is what fixes the
        dimension and this takes none, so the ``act`` paths keep that check.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import coordinax.transforms as cxfm

        For a stored matrix it is the field under a common name:

        >>> Rz = jnp.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        >>> bool(jnp.array_equal(cxfm.Rotate(Rz).matrix, Rz))
        True

        For a derived one it is the only way to see it:

        >>> cxfm.LorentzBoost([0.6, 0.0, 0.0]).matrix.round(2)
        Array([[1.25, 0.75, 0.  , 0.  ],
               [0.75, 1.25, 0.  , 0.  ],
               [0.  , 0.  , 1.  , 0.  ],
               [0.  , 0.  , 0.  , 1.  ]], dtype=float64)

        """
        return self._validate_square(self._raw_matrix)

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
        """Return the validated matrix for ``cart`` (constant; ``tau`` unused)."""
        del tau  # the matrix is constant; callers pass tau uniformly
        return self._validate_shape_match(self._validate_square(self._raw_matrix), cart)


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
    p_cart_out = _matmul_cdict(matrix, p_cart, comps_cart)
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
    n = op._validate_square(op._raw_matrix).shape[-1]

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


# ============================================================================
# pushforward — tangent geometry (shared by every linear transform)


@plum.dispatch
def pushforward(
    op: AbstractLinearTransform,
    tau: Any,
    v: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
) -> CDict:
    r"""Frozen-$\tau$ pushforward of tangent data under a linear map: $v \mapsto M v$.

    A linear map has a *constant* Jacobian equal to its matrix ``M``, so in the
    canonical Cartesian chart the pushforward is simply ``M v`` and needs no base
    point ``at`` (matching `Rotate`). For a non-Cartesian chart the tangent is
    pushed through the chart Jacobian (which does require ``at``), ``M`` is applied
    in Cartesian, then pulled back.

    This overload is the frozen-$\tau$ rule. For an order $\ge 1$ act on a
    *time-dependent* linear map, the ``act``-level router (see ``prolong``, which
    branches on ``is_time_dependent``) dispatches to the generic prolongation
    instead — that path adds the ``dot(M)`` term and requires the jet anchors —
    so time-dependent maps do not reach this frozen-$\tau$ overload via ``act``.

    Examples
    --------
    Scale a Cartesian velocity vector (no base point needed):

    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.transforms as cxfm

    >>> op = cxfm.Scale.from_factors([2.0, 3.0, 1.0])
    >>> v = {"x": u.Q(1.0, "m/s"), "y": u.Q(1.0, "m/s"), "z": u.Q(1.0, "m/s")}
    >>> out = cxfm.act(op, None, v, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel)
    >>> jnp.stack([out[c].to_value("m/s") for c in ("x", "y", "z")]).round(3)
    Array([2., 3., 1.], dtype=float64)

    """
    # The tangent must carry exactly the chart's components (a clear error
    # instead of a raw KeyError when packing); an anchor, if given, must match.
    ref = f"do not match the chart's {sorted(chart.components)}"
    for name, d in (("tangent", v), *(() if at is None else (("base point", at),))):
        pre = f"pushforward({type(op).__name__}, ...): the {name} components "
        require_matching_keys(d, chart.components, pre + ref)

    cart = chart.cartesian
    matrix = op._matrix(cart, tau)
    comps_cart = cart.components

    # Flat chart: the Jacobian is the identity, so M acts directly, no base point.
    if is_flat_chart(chart):
        return _matmul_cdict(matrix, v, comps_cart)

    # Non-flat: push the tangent through the chart Jacobian at `at`, apply M in
    # Cartesian, then pull back (anchoring the inverse Jacobian at M @ at).
    if at is None:
        msg = (
            f"pushforward({type(op).__name__}, ..., {rep!r}) on a "
            f"non-Cartesian chart ({chart!r}) requires 'at' (base point in "
            "chart coords) so the Jacobian pushforward can be evaluated."
        )
        raise TypeError(msg)
    at_cart = cxc.pt_map(at, chart, cart, usys=usys)
    p_cart = cxr.tangent_map(v, chart, rep, cart, at=at, usys=usys)  # ty: ignore[missing-argument]
    p_cart_out = _matmul_cdict(matrix, p_cart, comps_cart)
    at_out = _matmul_cdict(matrix, at_cart, comps_cart)
    return cxr.tangent_map(p_cart_out, cart, rep, chart, at=at_out, usys=usys)  # ty: ignore[missing-argument]


@plum.dispatch
def pushforward(
    op: AbstractLinearTransform,
    tau: Any,
    v: CDict,
    chart: cxc.AbstractCartesianProductChart,
    rep: cxr.Representation,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
) -> CDict:
    """Pushforward tangent data factorwise on a Cartesian-product chart.

    Mirrors the point action: the operator matrix is applied only to factors
    whose Cartesian dimension matches the matrix size (e.g. a 3x3 `Scale` acts on
    each `Cart3D` factor of a 6D phase-space chart); other factors pass through.

    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.transforms as cxfm

    >>> ps = cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q", "p"))
    >>> op = cxfm.Scale.from_factors([2.0, 3.0, 4.0])
    >>> v = {f"{f}.{c}": u.Q(1.0, "m/s") for f in ("q", "p") for c in "xyz"}
    >>> out = cxfm.act(op, None, v, ps, cxr.tangent_geom, cxr.coord_vel)
    >>> [out[k].value.round(3) for k in ("q.x", "q.y", "q.z", "p.x", "p.y", "p.z")]
    [Array(2., dtype=float64), Array(3., dtype=float64), Array(4., dtype=float64),
     Array(2., dtype=float64), Array(3., dtype=float64), Array(4., dtype=float64)]

    """
    # Validate the full product-chart keys up front (a clear error instead of a
    # raw KeyError from split_components), mirroring the non-product overload.
    ref = f"do not match the chart's {sorted(chart.components)}"
    for name, d in (("tangent", v), *(() if at is None else (("base point", at),))):
        pre = f"pushforward({type(op).__name__}, ...): the {name} components "
        require_matching_keys(d, chart.components, pre + ref)

    n = op._validate_square(op._raw_matrix).shape[-1]
    parts = chart.split_components(v)
    at_parts = (
        chart.split_components(at) if at is not None else [None] * len(chart.factors)
    )

    def _maybe(
        factor_chart: cxc.AbstractChart[Any, Any, Any],
        part: CDict,
        at_part: CDict | None,
        /,
    ) -> CDict:
        cart = factor_chart.cartesian
        if cart.ndim != n or len(cart.components) != n:
            return part
        return cast(
            "CDict",
            cxfmapi.pushforward(
                op, tau, part, factor_chart, rep, at=at_part, usys=usys
            ),
        )

    mapped = tuple(
        _maybe(f, p, a) for f, p, a in zip(chart.factors, parts, at_parts, strict=True)
    )
    return chart.merge_components(mapped)
