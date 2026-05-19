"""Galilean coordinate reflections."""

__all__ = ("Reflect",)


from jaxtyping import Array, Shaped
from typing import Any, Final, TypeAlias, cast, final

import jax.tree as jtu
import plum
from jax.typing import ArrayLike

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

HMatrix: TypeAlias = Shaped[Array, " N N"]

_MSG_H_SHAPE: Final = (
    "Reflect requires a square reflection matrix; got shape={shape!r}."
)

_MSG_H_X_SHAPE_MISMATCH: Final = (
    "Reflect() requires the chart's canonical Cartesian chart "
    "to have dimension matching the reflection matrix. "
    "Got H.shape={H.shape} and cartesian_chart={cart.__class__.__name__} "
    "with ndim={cart.ndim!r}."
)

_MSG_ZERO_NORMAL: Final = "Reflect.from_normal requires a nonzero normal vector."


@final
class Reflect(AbstractTransform):
    r"""Operator for Euclidean hyperplane reflections.

    A reflection across the hyperplane orthogonal to a nonzero normal vector $n$
    acts on Cartesian coordinates by the Householder matrix

    $$ H_n = I - 2\hat{n}\hat{n}^T, $$

    where $ \hat{n} = n / \lVert n \rVert $.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    >>> op = cxfm.Reflect.from_normal([1.0, 0.0, 0.0])
    >>> op.H
    Array([[-1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]], dtype=float64)

    >>> q = u.Q([1.0, 2.0, 3.0], "km")
    >>> cxfm.act(op, None, q)
    Q([-1.,  2.,  3.], 'km')

    """

    H: HMatrix
    """The reflection matrix."""

    @classmethod
    def groups(cls) -> frozenset[type]:
        """Return the groups to which this map belongs."""
        del cls
        return frozenset((groups.OrthogonalGroup, groups.DiffeomorphismGroup))

    def __init__(self, H: Any) -> None:
        object.__setattr__(self, "H", jnp.asarray(H))

    @classmethod
    def from_normal(cls: type["Reflect"], normal: Any, /) -> "Reflect":
        """Construct a Householder reflection from a hyperplane normal."""
        n = jnp.asarray(normal)
        if n.ndim != 1:
            msg = (
                f"Reflect.from_normal requires a vector normal; got shape={n.shape!r}."
            )
            raise ValueError(msg)

        norm = jnp.linalg.norm(n)
        if bool(jnp.allclose(norm, 0)):
            raise ValueError(_MSG_ZERO_NORMAL)

        n_hat = n / norm
        H = jnp.eye(n.shape[0], dtype=n_hat.dtype) - 2 * jnp.outer(n_hat, n_hat)
        return cls(H)

    @property
    def inverse(self) -> "Reflect":
        """The inverse of a reflection is the reflection itself."""
        return self

    @staticmethod
    def _validate_square(H: HasShape, /) -> HMatrix:
        shape = H.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(_MSG_H_SHAPE.format(shape=shape))
        return cast("HMatrix", H)

    @staticmethod
    def _validate_shape_match(
        H: HMatrix, cart: cxc.AbstractChart[Any, Any, Any], /
    ) -> HMatrix:
        n = H.shape[0]
        if cart.ndim != n or len(cart.components) != n:
            raise ValueError(_MSG_H_X_SHAPE_MISMATCH.format(H=H, cart=cart))
        return H

    def _get_H(self, cart: cxc.AbstractChart[Any, Any, Any], /) -> HMatrix:
        H = self._validate_square(self.H)
        return self._validate_shape_match(H, cart)


@Reflect.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Reflect], obj: Reflect, /) -> Reflect:
    """Construct a Reflect from another Reflect."""
    return obj


@Reflect.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Reflect], obj: AbcQ, /) -> Reflect:
    """Construct a Reflect from a dimensionless quantity matrix."""
    return cls(u.ustrip("", obj))


@Reflect.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Reflect], obj: ArrayLike, /) -> Reflect:
    """Construct a Reflect from an array matrix."""
    return cls(obj)


@plum.dispatch
def simplify(op: Reflect, /, **kw: Any) -> AbstractTransform:
    """Simplify a reflection, collapsing the identity matrix when present."""
    if jnp.allclose(op.H, jnp.eye(op.H.shape[0], dtype=op.H.dtype), **kw):
        return identity
    return op


@plum.dispatch
def act(
    op: Reflect,
    tau: Any,
    x: ArrayLike,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> Array:
    """Apply Reflect to an Array(like) object."""
    del tau, kw

    x_arr = jnp.asarray(x)
    chart = cxc.guess_chart(x_arr)  # ty: ignore[invalid-assignment]
    if chart != chart.cartesian:
        msg = "act for Reflect with ArrayLike x requires a Cartesian chart."
        raise ValueError(msg)
    if rep != cxr.point:
        msg = "act for Reflect with ArrayLike x requires a point representation."
        raise TypeError(msg)

    H = op._get_H(chart)
    return jnp.einsum("ij,...j->...i", H, x_arr)


@plum.dispatch
def act(
    op: Reflect,
    tau: Any,
    x: AbcQ,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> AbcQ:
    """Apply Reflect to a PointGeometry-roled Quantity."""
    del tau, rep, kw

    cart = chart.cartesian
    if chart != cart:
        msg = (
            "act(Reflect, ..., Quantity) requires Cartesian components. "
            f"chart {type(chart).__name__} is not its cartesian_chart."
        )
        raise ValueError(msg)

    H = op._get_H(chart)
    return jnp.einsum("ij,...j->...i", H, x)  # ty: ignore[invalid-return-type]


@plum.dispatch
def act(
    op: Reflect,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    *,
    usys: OptUSys = None,
    **kw: Any,
) -> CDict:
    """Apply Reflect to a coordinate dictionary."""
    out = cxfmapi.act(op, tau, x, chart, rep.geom_kind, rep, usys=usys, **kw)
    return cast("CDict", out)


@plum.dispatch
def act(
    op: Reflect,
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
    """Apply a hyperplane reflection to a Point-valued coordinate dictionary."""
    del tau, geom, rep, kw

    cart = chart.cartesian
    comps_cart = cart.components
    H = op._get_H(cart)

    p_cart = cxc.pt_map(x, chart, cart, usys=usys)

    v, unit = pack_uniform_unit(p_cart, keys=comps_cart)  # ty: ignore[no-matching-overload]
    v_ref = jnp.einsum("ij,...j->...i", H, v)
    p_cart_ref = cxc.cdict(v_ref, unit, comps_cart)

    out = cxc.pt_map(p_cart_ref, cart, chart, usys=usys)
    return cast("CDict", out)


@plum.dispatch
def act(
    op: Reflect,
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
    """Apply a reflection factorwise on Cartesian-product charts."""
    del tau
    n = op._validate_square(op.H).shape[-1]

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

        out = cxfmapi.act(op, None, part, factor_chart, geom, rep, usys=usys, **ats)
        return cast("CDict", out)

    reflected_parts = tuple(
        _maybe(f, p, **jtu.map(lambda seq, i=i: seq[i], ats))
        for i, (f, p) in enumerate(zip(chart.factors, parts, strict=True))
    )
    return chart.merge_components(reflected_parts)
