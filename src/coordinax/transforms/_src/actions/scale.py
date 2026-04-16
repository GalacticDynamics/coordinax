"""Pure spatial scaling transform."""
# ruff: noqa: I001

__all__ = ("Scale",)


from typing import Any, Final, TypeAlias, cast, final

import jax.tree as jtu
import plum
from jax.typing import ArrayLike
from jaxtyping import Array, Shaped

import quaxed.numpy as jnp
import unxt as u
from unxt import AbstractQuantity as AbcQ

import coordinax.api.transforms as cxfmapi
import coordinax.charts as cxc
import coordinax.representations as cxr
from .base import AbstractTransform
from .custom_types import CDict, HasShape, OptUSys
from .identity import identity
from coordinax.internal import pack_uniform_unit
from coordinax.transforms._src import groups

SMatrix: TypeAlias = Shaped[Array, " N N"]

_MSG_S_SHAPE: Final = "Scale requires a square scaling matrix; got shape={shape!r}."
_MSG_SINGULAR: Final = "Scale matrix must be invertible."
_MSG_S_X_SHAPE_MISMATCH: Final = (
    "Scale() requires the chart's canonical Cartesian chart "
    "to have dimension matching the scaling matrix. "
    "Got S.shape={S.shape} and cartesian_chart={cart.__class__.__name__} "
    "with ndim={cart.ndim!r}."
)


@final
class Scale(AbstractTransform):
    r"""Operator for Cartesian linear scaling.

    A scaling transform applies

    $$
    x \mapsto Sx,
    $$

    where ``S`` is an invertible scaling matrix. The common case is diagonal
    anisotropic scaling with per-axis factors.

    """

    S: SMatrix
    """The scaling matrix."""

    @classmethod
    def groups(cls) -> frozenset[type]:
        """Return the groups to which this map belongs."""
        del cls
        return frozenset((groups.AffineGroup, groups.DiffeomorphismGroup))

    def __init__(self, S: Any) -> None:
        object.__setattr__(self, "S", jnp.asarray(S))

    @classmethod
    def from_factors(cls: type["Scale"], factors: Any, /) -> "Scale":
        """Construct a diagonal scaling transform from axis factors."""
        s = jnp.asarray(factors)
        if s.ndim != 1:
            msg = f"Scale.from_factors requires a vector; got shape={s.shape!r}."
            raise ValueError(msg)
        if bool(jnp.any(jnp.isclose(s, 0))):
            raise ValueError(_MSG_SINGULAR)
        return cls(jnp.diag(s))

    @property
    def inverse(self) -> "Scale":
        """Return the inverse scaling transform."""
        return type(self)(jnp.linalg.inv(self.S))

    @staticmethod
    def _validate_square(S: HasShape, /) -> SMatrix:
        shape = S.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(_MSG_S_SHAPE.format(shape=shape))
        return cast("SMatrix", S)

    @staticmethod
    def _validate_shape_match(
        S: SMatrix, cart: cxc.AbstractChart[Any, Any], /
    ) -> SMatrix:
        n = S.shape[0]
        if cart.ndim != n or len(cart.components) != n:
            raise ValueError(_MSG_S_X_SHAPE_MISMATCH.format(S=S, cart=cart))
        return S

    def _get_S(self, cart: cxc.AbstractChart[Any, Any], /) -> SMatrix:
        S = self._validate_square(self.S)
        return self._validate_shape_match(S, cart)


@Scale.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Scale], obj: Scale, /) -> Scale:
    """Construct a Scale from another Scale."""
    return obj


@Scale.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Scale], obj: AbcQ, /) -> Scale:
    """Construct a Scale from a dimensionless quantity matrix."""
    return cls(u.ustrip("", obj))


@Scale.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Scale], obj: ArrayLike, /) -> Scale:
    """Construct a Scale from an array matrix."""
    return cls(obj)


@plum.dispatch
def simplify(op: Scale, /, **kw: Any) -> AbstractTransform:
    """Simplify a scaling transform to identity when matrix is identity."""
    if jnp.allclose(op.S, jnp.eye(op.S.shape[0], dtype=op.S.dtype), **kw):
        return identity
    return op


@plum.dispatch
def act(
    op: Scale,
    tau: Any,
    x: ArrayLike,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> Array:
    """Apply Scale to an Array(like) object."""
    del tau, kw

    x_arr = jnp.asarray(x)
    chart = cxc.guess_chart(x_arr)  # ty: ignore[invalid-assignment]
    if chart != chart.cartesian:
        msg = "act for Scale with ArrayLike x requires a Cartesian chart."
        raise ValueError(msg)
    if rep != cxr.point:
        msg = "act for Scale with ArrayLike x requires a point representation."
        raise TypeError(msg)

    S = op._get_S(chart)
    return jnp.einsum("ij,...j->...i", S, x_arr)


@plum.dispatch
def act(
    op: Scale,
    tau: Any,
    x: AbcQ,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> AbcQ:
    """Apply Scale to a PointGeometry-roled Quantity."""
    del tau, rep, kw

    cart = chart.cartesian
    if chart != cart:
        msg = (
            "act(Scale, ..., Quantity) requires Cartesian components. "
            f"chart {type(chart).__name__} is not its cartesian_chart."
        )
        raise ValueError(msg)

    S = op._get_S(chart)
    return jnp.einsum("ij,...j->...i", S, x)  # ty: ignore[invalid-return-type]


@plum.dispatch
def act(
    op: Scale,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    *,
    usys: OptUSys = None,
    **kw: Any,
) -> CDict:
    """Apply Scale to a coordinate dictionary."""
    out = cxfmapi.act(op, tau, x, chart, rep.geom_kind, rep, usys=usys, **kw)
    return cast("CDict", out)


@plum.dispatch
def act(
    op: Scale,
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
    """Apply Scale to a Point-valued coordinate dictionary."""
    del tau, geom, rep, kw

    cart = chart.cartesian
    comps_cart = cart.components
    S = op._get_S(cart)

    p_cart = cxc.pt_map(x, chart, cart, usys=usys)

    v, unit = pack_uniform_unit(p_cart, keys=comps_cart)  # ty: ignore[no-matching-overload]
    v_scaled = jnp.einsum("ij,...j->...i", S, v)
    p_cart_scaled = cxc.cdict(v_scaled, unit, comps_cart)

    out = cxc.pt_map(p_cart_scaled, cart, chart, usys=usys)
    return cast("CDict", out)


@plum.dispatch
def act(
    op: Scale,
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
    """Apply a scaling factorwise on Cartesian-product charts."""
    del tau
    n = op._validate_square(op.S).shape[-1]

    n_factors = len(chart.factors)
    parts = chart.split_components(x)
    ats = {
        k: chart.split_components(v) if v is not None else [None] * n_factors
        for k, v in kw.items()
        if k.startswith("at")
    }

    def _maybe(
        factor_chart: cxc.AbstractChart[Any, Any], part: CDict, /, **ats: CDict | None
    ) -> CDict:
        cart = factor_chart.cartesian
        if cart.ndim != n or len(cart.components) != n:
            return part

        out = cxfmapi.act(op, None, part, factor_chart, geom, rep, usys=usys, **ats)
        return cast("CDict", out)

    scaled_parts = tuple(
        _maybe(f, p, **jtu.map(lambda seq, i=i: seq[i], ats))
        for i, (f, p) in enumerate(zip(chart.factors, parts, strict=True))
    )
    return chart.merge_components(scaled_parts)
