"""Affine transform: one kernel for ``x -> A x + b``."""
# ruff: noqa: I001

__all__ = ("Affine",)

from typing import Any, TypeAlias, cast, final

import equinox as eqx
import plum
from jaxtyping import Array, Shaped

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
from .add import AbstractAdd
from .base import AbstractTransform
from .identity import identity
from .composed import _merge, simplify
from .linear import AbstractLinearTransform
from .utils import is_flat_chart
from coordinax._src.custom_types import OptUSys
from coordinax.internal import pack_uniform_unit
from coordinax.transforms._src import groups
from coordinax.transforms._src.groups import AbstractTransformGroup
from coordinaxs.api.custom_types import CDict
import coordinaxs.api.transforms as cxfmapi

AMatrix: TypeAlias = Shaped[Array, " N N"]


@final
class Affine(AbstractTransform):
    r"""A Cartesian affine map applied as a single kernel.

    $$
    x \mapsto A x + b
    $$

    A chain of affine operators is already expressible as `Composed`, but each
    element there costs its own chart round-trip and kernel launch. Composing
    them into one $(A, b)$ pair collapses that to a single einsum-plus-add:
    ICRS to Galactocentric is `Rotate | Translate | Rotate | Translate`, four
    applications, and the two `Rotate` operators are not adjacent, so pairwise
    merging cannot reach them.

    Like `~coordinax.transforms.Linear`, this names no structure of its own, so
    it carries the group it belongs to as a field rather than declaring one:
    fusing a rotation with a translation is still a Euclidean isometry, and
    reporting merely `AffineGroup` would discard that.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.transforms as cxfm

    A rotation followed by a translation, as one operator:

    >>> R = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> T = cxfm.Translate.from_([1.0, 0.0, 0.0], "km")
    >>> op = cxfm.simplify(R | T)
    >>> isinstance(op, cxfm.Affine)
    True

    It agrees with the chain it replaced:

    >>> p = cx.Point.from_([1.0, 0.0, 0.0], "km")
    >>> op(p)["x"].round(6), (R | T)(p)["x"].round(6)
    (Q(1., 'km'), Q(1., 'km'))

    """

    A: AMatrix
    """The linear part."""

    b: CDict
    """The offset, in ``chart`` components."""

    chart: cxc.AbstractChart = eqx.field(static=True)
    """The Cartesian chart ``A`` and ``b`` are expressed in."""

    group: type[AbstractTransformGroup] = eqx.field(
        static=True, default=groups.AffineGroup
    )
    """The tightest group this map is known to belong to."""

    def groups(self) -> frozenset[type]:
        """Return the groups to which this map belongs."""
        return frozenset((self.group, groups.DiffeomorphismGroup))

    @property
    def inverse(self) -> "Affine":
        r"""The inverse map, $x \mapsto A^{-1}(x - b)$.

        A group is closed under inversion, so the group travels unchanged.
        """
        inv = jnp.linalg.inv(self.A)
        b_val, b_unit = pack_uniform_unit(
            cast("CDict", self.b), keys=self.chart.components
        )
        neg = jnp.einsum("ij,...j->...i", inv, -b_val)
        return Affine(
            inv,
            cast("CDict", cxc.cdict(neg, b_unit, self.chart.components)),
            self.chart,
            self.group,
        )


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
    """Redispatch a CDict to the geometry-specific implementation."""
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
    """Apply ``A x + b`` in one kernel, in the chart's Cartesian components."""
    del tau, geom, rep, kw

    cart = chart.cartesian
    comps = cart.components
    p_cart = cxc.pt_map(x, chart, cart, usys=usys)

    x_val, x_unit = pack_uniform_unit(cast("CDict", p_cart), keys=comps)
    b_val, b_unit = pack_uniform_unit(cast("CDict", op.b), keys=comps)
    # The offset is stored in whatever unit it was built with; put it in the
    # data's unit rather than assuming they agree.
    if b_unit is not None and x_unit is not None and b_unit != x_unit:
        b_val = cast("Array", u.ustrip(x_unit, u.Quantity(b_val, b_unit)))
    out_val = jnp.einsum("ij,...j->...i", op.A, x_val) + b_val

    out_cart = cxc.cdict(out_val, x_unit, comps)
    return cast("CDict", cxc.pt_map(out_cart, cart, chart, usys=usys))


@plum.dispatch
def pushforward(
    op: Affine,
    tau: Any,
    v: CDict,
    chart: cxc.AbstractChart,
    rep: Any,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
) -> CDict:
    r"""Push a tangent vector forward: the offset differentiates away.

    $d(Ax + b) = A\,dx$, so a tangent sees only the linear part.

    That does *not* make the base point unnecessary. On a flat chart the chart
    Jacobian is the identity and $A$ applies directly. On a non-flat one the
    tangent must go through the chart Jacobian at ``at``, and the inverse
    Jacobian on the way back has to be anchored at the *image* of the base
    point -- which for an affine map is $A\,\mathrm{at} + b$, not
    $A\,\mathrm{at}$. Mapping the tangent with `pt_map` instead would treat it
    as a point: on `~coordinax.charts.sph3d` that tries to read ``rad/s`` as an
    angle and raises.
    """
    del tau

    cart = chart.cartesian
    comps = cart.components

    if is_flat_chart(chart):
        v_cart = cxc.pt_map(v, chart, cart, usys=usys)
        v_val, v_unit = pack_uniform_unit(cast("CDict", v_cart), keys=comps)
        out = jnp.einsum("ij,...j->...i", op.A, v_val)
        out_cart = cxc.cdict(out, v_unit, comps)
        return cast("CDict", cxc.pt_map(out_cart, cart, chart, usys=usys))

    if at is None:
        msg = (
            f"pushforward({type(op).__name__}, ..., {rep!r}) on a non-Cartesian "
            f"chart ({chart!r}) requires 'at' (base point in chart coords) so "
            "the Jacobian pushforward can be evaluated."
        )
        raise TypeError(msg)

    at_cart = cxc.pt_map(at, chart, cart, usys=usys)
    p_cart = cxr.tangent_map(v, chart, rep, cart, at=at, usys=usys)  # ty: ignore[missing-argument]
    p_val, p_unit = pack_uniform_unit(p_cart, keys=comps)
    out_cart = cxc.cdict(jnp.einsum("ij,...j->...i", op.A, p_val), p_unit, comps)

    at_val, at_unit = pack_uniform_unit(cast("CDict", at_cart), keys=comps)
    mapped = cast(
        "CDict", cxc.cdict(jnp.einsum("ij,...j->...i", op.A, at_val), at_unit, comps)
    )
    at_out = {k: mapped[k] + op.b[k] for k in comps}
    return cxr.tangent_map(out_cart, cart, rep, chart, at=at_out, usys=usys)  # ty: ignore[missing-argument]


# ============================================================================
# Collapsing a run of affine operators
#
# `simplify(Composed)` merges adjacent pairs left-to-right and re-simplifies
# after each merge, so folding the *pairs* below is enough to collapse a whole
# run: `Rotate | Translate | Rotate | Translate` becomes one `Affine` without a
# separate maximal-run pass.


def _affine_parts(
    op: AbstractTransform, /
) -> tuple[AMatrix, CDict, cxc.AbstractChart, type[AbstractTransformGroup]] | None:
    """Return ``(A, b, chart, group)`` for *op*, or `None` if it is not affine.

    Membership is decided on the declared lattice via `groups.is_subgroup`, not
    `issubclass` -- see its docstring. That is what keeps a `LorentzBoost` out:
    it subclasses `OrthogonalGroup` but declares `PoincareGroup`, and folding a
    4x4 spacetime map into a chain of 3x3 spatial ones would be a shape error
    dressed as an optimisation.
    """
    grp = groups.most_specific_group(op.groups())
    if not groups.is_subgroup(grp, groups.AffineGroup):
        return None

    if isinstance(op, Affine):
        return op.A, op.b, op.chart, grp

    if isinstance(op, AbstractAdd):
        # Not every additive operator is a *static point* offset, and the two
        # that are not would fuse into nonsense:
        #
        # - `Boost` acts as `x + dv*tau`, so its offset is tau-dependent and
        #   has no place in a constant `b`. It declares no `semantic_kind` at
        #   all, which is how it is spotted here.
        # - `Translate(semantic_kind=vel)` and friends are fibre-only (ladder
        #   order k >= 1): they move velocities, not points, so folding them
        #   into the point map would move the wrong thing.
        #
        # ICRS <-> Galactocentric carries both a spatial `Translate` in kpc and
        # a `Boost` in km/s; fusing them tried to convert one into the other.
        kind = getattr(op, "semantic_kind", None)
        if kind is None or getattr(kind, "order", 1) != 0:
            return None
        chart = cast("cxc.AbstractChart", op.chart)
        n = len(chart.components)
        return jnp.eye(n), cast("CDict", op.delta), chart, grp

    if isinstance(op, AbstractLinearTransform):
        matrix = op.matrix
        n = matrix.shape[-1]
        comps = _CART_COMPONENTS.get(n)
        if comps is None:
            # No Cartesian chart of this dimension to name the offset in, so
            # there is no `A x + b` to fold into. Decline rather than raise: a
            # pair that does not combine is `_merge`'s ordinary answer, and a
            # `KeyError` out of `simplify` would be a crash, not a refusal.
            return None
        cart = cast("cxc.AbstractChart", cxc.guess_chart(frozenset(comps)))
        zero = cast("CDict", cxc.cdict(jnp.zeros(n), None, cart.components))
        return matrix, zero, cart, grp

    return None


# The Cartesian charts an offset can be expressed in. A linear map of any other
# dimension -- a 4x4 spacetime map, say -- has no entry here and simply does not
# fold; see `_affine_parts`.
_CART_COMPONENTS: dict[int, tuple[str, ...]] = {
    1: ("x",),
    2: ("x", "y"),
    3: ("x", "y", "z"),
}


def _compose_affine(
    a: AbstractTransform, b: AbstractTransform, /
) -> AbstractTransform | None:
    r"""Fold ``a`` then ``b`` into one `Affine`, or decline.

    With ``a`` applied first, $x \mapsto A_b(A_a x + b_a) + b_b$, so

    $$
    A = A_b A_a, \qquad b = A_b b_a + b_b.
    $$
    """
    pa, pb = _affine_parts(a), _affine_parts(b)
    if pa is None or pb is None:
        return None

    A_a, b_a, chart_a, g_a = pa
    A_b, b_b, chart_b, g_b = pb

    # Shapes are static under `jit`, so this check traces.
    if A_a.shape != A_b.shape or chart_a.components != chart_b.components:
        return None

    ba_val, unit_a = pack_uniform_unit(cast("CDict", b_a), keys=chart_a.components)
    bb_val, unit_b = pack_uniform_unit(cast("CDict", b_b), keys=chart_b.components)

    # A purely linear operator contributes a *unitless* zero offset, so the
    # shared unit is whichever operand actually carries one. When both do, the
    # second is converted into the first rather than assumed compatible.
    unit = unit_a if unit_a is not None else unit_b
    if unit_a is not None and unit_b is not None and unit_b != unit_a:
        bb_val = cast("Array", u.ustrip(unit_a, u.Quantity(bb_val, unit_b)))

    offset = jnp.einsum("ij,...j->...i", A_b, ba_val) + bb_val

    return Affine(
        A_b @ A_a,
        cast("CDict", cxc.cdict(offset, unit, chart_a.components)),
        chart_a,
        groups.least_common_supergroup((g_a, g_b)),
    )


@plum.dispatch.multi(
    (AbstractLinearTransform, AbstractAdd),
    (AbstractAdd, AbstractLinearTransform),
    (Affine, AbstractTransform),
    (AbstractTransform, Affine),
    # `(Affine, Affine)` explicitly: the two mixed signatures above both match
    # it and neither is the more specific, so plum calls it ambiguous rather
    # than picking one. It arises when a chain folds to an `Affine` on each
    # side of an `Identity` that is then stripped -- and when one is composed
    # with its own inverse.
    (Affine, Affine),
)
def _merge(a: AbstractTransform, b: AbstractTransform, /) -> AbstractTransform | None:
    """Fold an adjacent affine pair into a single `Affine`."""
    return _compose_affine(a, b)


@plum.dispatch
def simplify(op: Affine, /, *, approx: bool = True, **kw: Any) -> AbstractTransform:
    """Collapse to `Identity` when the map is one; otherwise keep the fusion.

    Both checks inspect values, so both are skipped when ``approx=False`` --
    the same trace-safety contract the sibling operators honour.
    """
    if not approx:
        return op

    n = op.A.shape[-1]
    b_val, _ = pack_uniform_unit(cast("CDict", op.b), keys=op.chart.components)
    if jnp.allclose(op.A, jnp.eye(n), **kw) and jnp.allclose(b_val, 0.0, **kw):
        return identity
    return op
