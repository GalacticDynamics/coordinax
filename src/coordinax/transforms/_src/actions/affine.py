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
    applications, and the `Rotate`s are not adjacent so pairwise merging cannot
    reach them.

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

    $d(Ax + b) = A\,dx$, so a tangent transforms by the linear part alone and
    needs no base point.
    """
    del tau, rep, at

    cart = chart.cartesian
    comps = cart.components
    v_cart = cxc.pt_map(v, chart, cart, usys=usys)
    v_val, v_unit = pack_uniform_unit(cast("CDict", v_cart), keys=comps)
    out = jnp.einsum("ij,...j->...i", op.A, v_val)
    out_cart = cxc.cdict(out, v_unit, comps)
    return cast("CDict", cxc.pt_map(out_cart, cart, chart, usys=usys))


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
        #   has no place in a constant `b`. Its delta is a bare velocity
        #   `Quantity`, not a component dict, which is how it is spotted here.
        # - `Translate(semantic_kind=vel)` and friends are fibre-only (ladder
        #   order k >= 1): they move velocities, not points, so folding them
        #   into the point map would move the wrong thing.
        #
        # ICRS <-> Galactocentric carries both a spatial `Translate` in kpc and
        # a `Boost` in km/s; fusing them tried to convert one into the other.
        kind = getattr(op, "semantic_kind", None)
        if kind is None or getattr(kind, "order", 1) != 0:
            return None
        if not isinstance(op.delta, dict):
            return None
        chart = cast("cxc.AbstractChart", op.chart)
        n = len(chart.components)
        return jnp.eye(n), cast("CDict", op.delta), chart, grp

    if isinstance(op, AbstractLinearTransform):
        matrix = op.matrix
        n = matrix.shape[-1]
        cart = cast(
            "cxc.AbstractChart", cxc.guess_chart(frozenset(_CART_COMPONENTS[n]))
        )
        zero = cast("CDict", cxc.cdict(jnp.zeros(n), None, cart.components))
        return matrix, zero, cart, grp

    return None


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
