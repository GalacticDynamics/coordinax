"""Affine transform: one kernel for a linear map plus an offset."""
# ruff: noqa: I001

__all__ = ("Affine",)


from typing import Any, TypeAlias, cast, final

import equinox as eqx
import plum
from jaxtyping import Array, Shaped

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinaxs.api.transforms as cxfmapi

from .base import AbstractTransform
from .custom_types import CDict, OptUSys
from .identity import identity
from .linear import AbstractLinearTransform
from .utils import is_flat_chart, require_matching_keys
from coordinax.internal import pack_uniform_unit
from .translate import Translate
from coordinax.transforms._src import groups

AMatrix: TypeAlias = Shaped[Array, " N N"]


def _apply(matrix: Any, d: CDict, comps: tuple[str, ...], /) -> CDict:
    """Contract ``matrix`` with a Cartesian cdict, packed into a shared unit.

    `linear.py` has an equivalent, but it routes through the *operator* so a
    subclass can override the contraction. `Affine` is not one of those
    operators, and `_merge` applies a matrix belonging to a neighbour, so a
    plain function is both simpler and correct here.
    """
    v, unit = pack_uniform_unit(d, keys=comps)
    return cast("CDict", cxc.cdict(jnp.einsum("ij,...j->...i", matrix, v), unit, comps))


@final
class Affine(AbstractTransform):
    r"""Operator for a Cartesian affine map, applied as one kernel.

    An affine transform applies

    $$
    x \mapsto A x + b,
    $$

    with $A$ an invertible matrix and $b$ an offset.

    This exists for *composition*, not for expressiveness: a rotation followed
    by a translation is already sayable as ``R | T``. The point is that a chain
    of affine operators -- the shape every frame transition takes, e.g.
    ``Rotate | Translate | Rotate | Translate`` -- collapses into a single
    `Affine`, so the whole chain costs one chart round-trip, one matrix
    contraction and one addition, rather than one of each per operator.

    The adjacent-pair peephole cannot do this on its own: in that chain the two
    `Rotate` operators are not neighbours, so there is no pair to merge.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxfm

    An interleaved affine chain collapses to one operator:

    >>> R = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> T = cxfm.Translate.from_([1.0, 0.0, 0.0], "m")
    >>> cxfm.simplify(R | T | R | T)
    Affine(...)

    and it agrees with the chain it replaces:

    >>> p = cx.Point.from_(u.Q(jnp.asarray([1.0, 2.0, 3.0]), "m"), cxc.cart3d)
    >>> chain, fused = R | T | R | T, cxfm.simplify(R | T | R | T)
    >>> bool(jnp.allclose(chain(p)["x"].ustrip("m"), fused(p)["x"].ustrip("m")))
    True

    """

    A: AMatrix
    """The linear part, acting on the chart's Cartesian components."""

    delta: CDict
    """The offset $b$, in ``chart``'s Cartesian components."""

    chart: cxc.AbstractChart = eqx.field(static=True)
    """The Cartesian chart ``delta`` is expressed in (static)."""

    def __init__(self, A: Any, delta: CDict, chart: cxc.AbstractChart) -> None:
        object.__setattr__(self, "A", jnp.asarray(A))
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "chart", chart)

    @classmethod
    def groups(cls) -> frozenset[type]:
        """Return the groups to which this map belongs."""
        del cls
        return frozenset((groups.AffineGroup, groups.DiffeomorphismGroup))

    @property
    def inverse(self) -> "Affine":
        r"""Return the inverse affine map, $x \mapsto A^{-1}(x - b)$.

        The offset is carried through the inverse linear part, not merely
        negated: undoing $Ax + b$ means subtracting $b$ *before* applying
        $A^{-1}$.
        """
        inv = jnp.linalg.inv(self.A)
        keys = tuple(self.delta)
        neg = {k: -self.delta[k] for k in keys}
        return Affine(inv, _apply(inv, neg, keys), self.chart)


# ============================================================================
# act


@plum.dispatch
def act(
    op: Affine,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    *,
    usys: OptUSys = None,
    **kw: Any,
) -> CDict:
    """Redispatch a CDict to the geometry-specific implementation.

    Mirrors the linear base: the geometry, not the representation, decides
    whether this is a point action or a pushforward.
    """
    out = cxfmapi.act(op, tau, x, chart, rep.geom_kind, rep, usys=usys, **kw)
    return cast("CDict", out)


@plum.dispatch
def act(
    op: Affine,
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
    """Apply the affine map to a point.

    One kernel: to Cartesian once, contract, add, and back once -- however many
    operators were fused to build this one.
    """
    del tau, geom, rep, kw

    cart = chart.cartesian
    comps = cart.components
    p_cart = cxc.pt_map(x, chart, cart, usys=usys)
    mapped = _apply(op.A, p_cart, comps)
    shifted = {k: mapped[k] + op.delta[k] for k in comps}
    return cast("CDict", cxc.pt_map(shifted, cart, chart, usys=usys))


# ============================================================================
# pushforward -- tangent geometry


@plum.dispatch
def pushforward(
    op: Affine,
    tau: Any,
    v: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
) -> CDict:
    r"""Push a tangent vector through the affine map: $v \mapsto A v$.

    The offset does not appear: the Jacobian of $x \mapsto Ax + b$ is $A$, so
    a *tangent* sees only the linear part. The offset does still matter on a
    non-flat chart, because the pulled-back Jacobian must be anchored at the
    image of the base point -- and that image is $A\,\mathrm{at} + b$, not
    $A\,\mathrm{at}$. Anchoring at the latter is the mistake this dispatch
    exists to avoid; delegating to the linear pushforward would make it.
    """
    del tau
    ref = f"do not match the chart's {sorted(chart.components)}"
    for name, d in (("tangent", v), *(() if at is None else (("base point", at),))):
        pre = f"pushforward({type(op).__name__}, ...): the {name} components "
        require_matching_keys(d, chart.components, pre + ref)

    cart = chart.cartesian
    comps = cart.components

    # Flat chart: the chart Jacobian is the identity, so A acts directly and no
    # base point is needed -- the offset is irrelevant to a tangent here.
    if is_flat_chart(chart):
        return _apply(op.A, v, comps)

    if at is None:
        msg = (
            f"pushforward({type(op).__name__}, ..., {rep!r}) on a non-Cartesian "
            f"chart ({chart!r}) requires 'at' (base point in chart coords) so "
            "the Jacobian pushforward can be evaluated."
        )
        raise TypeError(msg)

    at_cart = cxc.pt_map(at, chart, cart, usys=usys)
    p_cart = cxr.tangent_map(v, chart, rep, cart, at=at, usys=usys)  # ty: ignore[missing-argument]
    p_cart_out = _apply(op.A, p_cart, comps)
    mapped_at = _apply(op.A, at_cart, comps)
    at_out = {k: mapped_at[k] + op.delta[k] for k in comps}
    # `ty: ignore` as in `linear.py`: plum's `tangent_map` overload set is not
    # visible to the checker, which reads the four-positional form as missing
    # `to_rep`.
    return cxr.tangent_map(p_cart_out, cart, rep, chart, at=at_out, usys=usys)  # ty: ignore[missing-argument]


# ============================================================================
# simplify


@plum.dispatch
def simplify(op: Affine, /, *, approx: bool = True, **kw: Any) -> AbstractTransform:
    """Collapse a degenerate affine map back to a simpler operator.

    A zero offset is a pure linear map, and an identity linear part is a pure
    translation; both checks read values, so both are skipped when
    ``approx=False``.
    """
    from .general_linear import Linear  # noqa: PLC0415  (cycle: Linear -> Affine)

    if not approx:
        return op

    n = op.A.shape[0]
    # `ustrip(AllowValue, ...)` accepts a Quantity or a bare array alike, so
    # there is no need to sniff for `.value` -- the same hand-rolled unwrapping
    # that was removed from the manifolds code earlier.
    no_shift = all(
        bool(jnp.allclose(u.ustrip(AllowValue, u.unit_of(v) or "", v), 0.0, **kw))
        for v in op.delta.values()
    )
    is_eye = bool(jnp.allclose(op.A, jnp.eye(n, dtype=op.A.dtype), **kw))

    if is_eye and no_shift:
        return identity
    if no_shift:
        return simplify(Linear(op.A), approx=approx, **kw)
    if is_eye:
        return Translate(op.delta, op.chart)
    return op


# ============================================================================
# _merge -- absorbing neighbours, which is the whole point


def _parts(op: AbstractTransform, /) -> tuple[Any, CDict | None, Any] | None:
    """Read ``op`` as ``(A, b, chart)``, or `None` if it is not static affine.

    `None` for either part means "identity for that part": a linear operator
    contributes no offset, a translation no matrix.
    """
    if isinstance(op, Affine):
        return op.A, op.delta, op.chart
    if isinstance(op, AbstractLinearTransform):
        return op.matrix, None, None
    if isinstance(op, Translate):
        # Only a *position* offset is affine on points. A fibre kick
        # (`semantic_kind` of order >= 1) acts on the tangent, and a
        # left-adding one is a different map; neither belongs in `A x + b`.
        if op.semantic_kind.order != 0 or not op.right_add:
            return None
        if op.chart != op.chart.cartesian:
            return None  # a non-Cartesian offset is not componentwise
        return None, op.delta, op.chart
    return None


def _fuse(a: AbstractTransform, b: AbstractTransform, /) -> AbstractTransform | None:
    r"""Fuse two adjacent static affine operators into one `Affine`.

    With ``a`` applied first, $x \mapsto A_b(A_a x + b_a) + b_b$, so the fused
    parts are $A_b A_a$ and $A_b b_a + b_b$. The offset is pushed *through*
    the second matrix -- adding the two offsets would be wrong whenever
    $A_b \neq I$.

    Same-type pairs never reach here: `Rotate | Rotate` and the rest have their
    own, more specific rules, and plum prefers those.
    """
    pa, pb = _parts(a), _parts(b)
    if pa is None or pb is None:
        return None

    A_a, b_a, chart_a = pa
    A_b, b_b, chart_b = pb

    chart = chart_a if chart_a is not None else chart_b
    if chart is None:  # two pure linear maps: not ours, the linear rule has it
        return None
    if chart_a is not None and chart_b is not None and chart_a != chart_b:
        return None  # offsets in different charts do not add componentwise

    comps = chart.components
    n = (A_a if A_a is not None else A_b).shape[0]
    if len(comps) != n:
        return None

    A = (
        A_b @ A_a
        if (A_a is not None and A_b is not None)
        else (A_a if A_b is None else A_b)
    )

    # b = A_b @ b_a + b_b
    shifted = _apply(A_b, b_a, comps) if (b_a is not None and A_b is not None) else b_a
    if shifted is None:
        delta = b_b
    elif b_b is None:
        delta = shifted
    else:
        delta = {k: shifted[k] + b_b[k] for k in comps}

    return Affine(A if A is not None else jnp.eye(n), delta, chart)


# Registered as explicit pairs rather than one `Affine | AbstractLinearTransform
# | Translate` union on both sides. That union overlaps `_merge(AbstractAdd,
# AbstractAdd)` on a `(Translate, Translate)` call without being narrower than
# it -- `Translate` is an `AbstractAdd` but `Affine` is not -- so neither
# signature dominates and every such call raises `AmbiguousLookupError`.
#
# `(Translate, Translate)` is deliberately absent: two position offsets already
# merge into a `Translate`, which is a better answer than an `Affine` with an
# identity matrix.


@plum.dispatch
def _merge(a: Affine, b: Affine, /) -> AbstractTransform | None:
    """Fuse two affine maps."""
    return _fuse(a, b)


@plum.dispatch
def _merge(a: Affine, b: AbstractLinearTransform, /) -> AbstractTransform | None:
    """Absorb a following linear map."""
    return _fuse(a, b)


@plum.dispatch
def _merge(a: AbstractLinearTransform, b: Affine, /) -> AbstractTransform | None:
    """Absorb a preceding linear map."""
    return _fuse(a, b)


@plum.dispatch
def _merge(a: Affine, b: Translate, /) -> AbstractTransform | None:
    """Absorb a following translation."""
    return _fuse(a, b)


@plum.dispatch
def _merge(a: Translate, b: Affine, /) -> AbstractTransform | None:
    """Absorb a preceding translation."""
    return _fuse(a, b)


@plum.dispatch
def _merge(a: AbstractLinearTransform, b: Translate, /) -> AbstractTransform | None:
    """Seed an affine chain: a linear map followed by a translation."""
    return _fuse(a, b)


@plum.dispatch
def _merge(a: Translate, b: AbstractLinearTransform, /) -> AbstractTransform | None:
    """Fuse a translation then a linear map; the offset rides through."""
    return _fuse(a, b)
