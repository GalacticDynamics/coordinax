"""Abstract additive operator base class."""

__all__ = ("AbstractAdd",)

import functools as ft
from dataclasses import KW_ONLY, replace

from jaxtyping import ArrayLike
from typing import Any, Union, cast

import equinox as eqx
import jax.tree as jtu
import plum
import wadler_lindig as wl

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items
from unxt.quantity import AllowValue, is_any_quantity

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinaxs.api.transforms as cxfmapi
from .base import AbstractTransform
from .composed import Composed
from .composite import AbstractCompositeTransform
from .custom_types import CDict
from .identity import Identity, identity
from .prolong import (
    _MSG_JET_SLOT0_MISSING,
    prolong_jet,
    pushforward_generic,
    tau_derivative,
)
from .timedep import TimeDep
from .utils import is_componentwise_offset
from coordinax.internal import jax_scalar_handler, pos_named_objs

_MSG_CALLABLE_DELTA = (
    "{cls}.delta must be a component dict, not a callable. Time-dependent "
    "offsets are no longer expressed by passing a function: wrap a builder in "
    "coordinax.transforms.TimeDep instead, e.g. "
    "TimeDep(UniformTranslation(rate, chart=...)) for a uniform drift, or "
    "TimeDep.from_(lambda tau: {cls}(delta(tau), chart=...)) for an "
    "arbitrary one."
)


_MSG_COMPOSED_FIBRE_OFFSET = (
    "a TimeDep builder returned a composite containing a fibre offset "
    "({cls} with semantic_kind of order >= 1). That spelling is not "
    "supported: the fibre-offset ladder rule can only see an offset that is "
    "the whole materialized transform, so the offset would be silently "
    "dropped on tangent data. Put the TimeDep INSIDE the composition "
    "instead -- e.g. `static_part | TimeDep(kick_builder)` rather than "
    "`TimeDep(lambda tau: static_part | kick(tau))`."
)


class AbstractAdd(AbstractTransform):
    """Abstract base class for additive operators (Translate, Boost, etc.).

    Additive operators represent field-like quantities (displacements, velocity
    offsets, etc.) that can be combined via addition and negated.

    Common features:
    - Addition of two operators combines their offsets
    - Negation inverts the offset
    - Time-dependent offsets: wrap in `~coordinax.transforms.TimeDep`
    - Chart-aware representation
    """

    delta: CDict
    """The additive offset (displacement for Translate, velocity for Boost)."""

    chart: cxc.AbstractChart = eqx.field(static=True)
    """Chart in which the offset is expressed."""

    _: KW_ONLY

    right_add: bool = eqx.field(default=True, static=True)
    """Whether to add on the right (x + offset) or left (offset + x)."""

    def __check_init__(self) -> None:
        """Reject a callable ``delta`` at construction.

        ``delta`` is a plain `dict` field with no coercion, so a function would
        otherwise sail through ``__init__`` and only fail somewhere downstream.
        Runtime type-checking rejects it too, but that is enabled only under
        this project's pytest configuration — for library users this guard is
        the ONLY construction-time rejection, so it must not be removed on the
        grounds that the type annotation "already covers it".

        No doctest here: under the doctest runner the type-checker preempts
        this method, so an example would demonstrate the wrong error. See
        ``tests/unit/transforms/test_translate.py::TestCallableDeltaRejected``,
        which calls this method directly.
        """
        if callable(self.delta):
            raise TypeError(_MSG_CALLABLE_DELTA.format(cls=type(self).__name__))

    def _combine_offsets(self, other_offset: CDict) -> CDict:
        """Combine this offset with another via addition."""
        return jtu.map(jnp.add, self.delta, other_offset, is_leaf=is_any_quantity)

    def __neg__(self) -> "AbstractAdd":
        """Return negative of the operator."""
        return self.inverse

    @property
    def inverse(self) -> "AbstractAdd":
        """The inverse operator (negated offset).

        Examples
        --------
        >>> import coordinax.transforms as cxfm

        >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
        >>> shift.inverse
        Translate(
            {'x': Q(-1, 'km'), 'y': Q(-2, 'km'), 'z': Q(-3, 'km')},
            chart=Cart3D(M=Rn(3))
        )

        """
        inv = jtu.map(jnp.negative, self.delta, is_leaf=is_any_quantity)
        return replace(self, delta=inv)

    def __add__(self, other: object, /) -> Union["AbstractAdd", Composed]:
        """Combine two operators of the same type."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return replace(self, delta=self._combine_offsets(other.delta))

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(self, **kw: Any) -> wl.AbstractDoc:
        """Wadler-Lindig documentation for Translate operator."""
        # Set pdoc option defaults
        kw.setdefault("include_params", False)
        kw.setdefault("short_arrays", "compact")
        kw.setdefault("use_short_names", True)
        kw.setdefault("named_unit", False)

        # Build the fields
        fitems = cast("list[tuple[str, Any]]", field_items(self))
        kw = {**kw, "custom": jax_scalar_handler}
        docs = pos_named_objs(fitems, ["delta"], self.__dataclass_fields__, **kw)

        # Return the full doc
        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}("),
            docs=docs,
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=4,
        )

    def __repr__(self) -> str:
        """Return string representation of Add operator."""
        return wl.pformat(
            self.__pdoc__(
                short_arrays="compact",
                use_short_name=True,
                include_params=False,
                named_unit=False,
            ),
            width=80,
        )

    __str__ = __repr__


# ============================================================================
# Constructors


@AbstractAdd.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[AbstractAdd], obj: AbstractAdd, /) -> AbstractAdd:
    """Construct a AbstractAdd from another AbstractAdd.

    >>> import coordinax as cx
    >>> shift1 = cxfm.Translate.from_([1, 2, 3], "km")
    >>> cxfm.Translate.from_(shift1) is shift1
    True

    """
    if type(obj) is not cls:
        raise TypeError(f"Cannot construct {cls.__name__} from {type(obj).__name__}")
    return obj


@AbstractAdd.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[AbstractAdd], q: u.AbstractQuantity, /) -> AbstractAdd:
    """Construct an AbstractAdd subclass from a Quantity.

    >>> import unxt as u
    >>> import coordinax.transforms as cxfm
    >>> cxfm.Translate.from_(u.Q([1, 2, 3], "km"))
    Translate(
        {'x': Q(1, 'km'), 'y': Q(2, 'km'), 'z': Q(3, 'km')}, chart=Cart3D(M=Rn(3))
    )

    """
    chart = cxc.guess_chart(q)
    x = cxc.cdict(q, chart)
    return cls(x, chart=chart)


@AbstractAdd.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[AbstractAdd], x: ArrayLike, unit: str) -> AbstractAdd:
    """Construct an Add operator from an array-like offset and unit.

    >>> import coordinax.transforms as cxfm
    >>> cxfm.Translate.from_([1, 2, 3], "km")
    Translate(
        {'x': Q(1, 'km'), 'y': Q(2, 'km'), 'z': Q(3, 'km')}, chart=Cart3D(M=Rn(3))
    )

    """
    return cls.from_(u.Q(x, unit))  # ty: ignore[invalid-return-type]


# ============================================================================
# act_jet


@ft.cache
def _slot_rep(m: int, /) -> Any:
    """Return the coordinate-basis representation for jet slot ``m`` (cached)."""
    if m == 0:
        return cxr.point
    kind: cxr.AbstractTangentSemanticKind = cxr.vel
    while kind.order < m:
        kind = kind.derivative()
    return cxr.Representation(cxr.tangent_geom, cxr.coord_basis, kind)


@plum.dispatch
def act_jet(
    op: AbstractAdd,
    tau: Any,
    jet: dict,
    chart: cxc.AbstractChart,
    /,
    *,
    usys: Any = None,
) -> dict:
    r"""Prolong an additive operator slot-wise.

    When the offset and the jet live in the same Cartesian-type (flat) chart
    (or the operator is a fibre-only offset, whose point action is the
    identity), the point Jacobian is the identity and the prolongation has
    no cross-slot coupling: each jet slot transforms
    independently by the operator's ladder rule (slot $m$ gains
    $d^{m-k}\delta/d\tau^{m-k}$ for the operator's ladder order $k$). This
    also makes fibre-only offsets (e.g. ``Translate(semantic_kind=vel)``) —
    which are invisible to the generic point-action prolongation — correct
    under ``act_jet``.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.transforms as cxfm

    >>> kick = cxfm.Translate(
    ...     {"x": u.Q(100.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
    ...     chart=cxc.cart3d, semantic_kind=cxr.vel,
    ... )
    >>> jet = {0: {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
    ...        1: {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")}}
    >>> out = cxfm.act_jet(kick, None, jet, cxc.cart3d)
    >>> out[0]["x"], out[1]["x"]
    (Q(1., 'm'), Q(101., 'm / s'))

    """
    if is_componentwise_offset(op, chart):
        return _prolong_slotwise(op, tau, jet, chart, usys=usys)

    # A point-active offset (ladder order 0) outside the flat matching case
    # is fully captured by the point action — use the generic prolongation.
    return prolong_jet(op, tau, jet, chart, usys=usys)


def _prolong_slotwise(
    op: AbstractTransform,
    tau: Any,
    jet: dict,
    chart: cxc.AbstractChart,
    /,
    *,
    usys: Any = None,
) -> dict:
    """Prolong slot-wise: each jet slot transforms by the ladder rule alone."""
    # The jet supplies the base point, so fibre kicks (k >= 1) work even
    # cross-chart: `act` pushes the offset through the chart Jacobian at
    # jet[0] when needed. The anchor is only passed for tangent slots:
    # slot 0 IS the point, so a strict point dispatch need not accept it.
    # Any tangent slot indexes jet[0], so require it explicitly here — a
    # bare KeyError would otherwise mask the same guard prolong_jet gives.
    if 0 not in jet and any(m != 0 for m in jet):
        raise TypeError(_MSG_JET_SLOT0_MISSING)
    return {
        m: cxfmapi.act(
            op,
            tau,
            slot,
            chart,
            _slot_rep(m),
            usys=usys,
            **({"at": jet[0]} if m else {}),
        )
        for m, slot in jet.items()
    }


# ============================================================================
# Time-dependent FIBRE offsets: `TimeDep` wrapping an additive kick.
#
# A fibre offset (ladder order k >= 1) has an IDENTITY point action, so the
# generic tangent funnel in prolong.py — which recovers time dependence by
# differentiating the point action — has nothing to differentiate and is
# provably blind to it (it would silently return the data unchanged). The
# rules below are the ladder rule that `translate.py` applies to a static
# fibre offset, re-homed for the `TimeDep` families that now carry all
# time dependence.
#
# THE PREDICATE IS LOAD-BEARING. It must fire ONLY for additive fibre offsets.
# Everything else — in particular every point-acting transform, where the
# funnel's differentiation supplies the dR/dtau and dgamma/dtau terms this
# whole design exists to capture — must fall through to the funnel unchanged.


@ft.cache
def _generic_tangent_act() -> Any:
    """Return the engine's generic tangent ``act`` (``prolong.py``), lazily.

    The `TimeDep` rule below dispatches strictly more specifically than the
    engine's, so a non-fibre `TimeDep` can only get the engine's behaviour by
    invoking it explicitly. Resolved on first call (and cached) because the
    registration does not exist at this module's import time.
    """
    return cxfmapi.act.invoke(
        AbstractTransform,
        object,  # the engine registers `tau: Any`; `invoke` wants a runtime type
        CDict,
        cxc.AbstractChart,
        cxr.TangentGeometry,
        cxr.Representation,
    )


def _ladder_order(op0: Any, /) -> int | None:
    r"""Ladder order $k$ of a materialized fibre offset, else `None`.

    Mirrors `~coordinax.transforms.is_componentwise_offset`'s notion of a
    fibre offset (the additive family's routing predicate): an `AbstractAdd`
    whose ``semantic_kind`` order is $k \\geq 1$. Order-0 additives are real
    translations with a non-identity point action and are NOT fibre offsets.

    A composite hiding a fibre offset is REJECTED rather than silently routed
    to the funnel; see `_reject_composed_fibre_offset`.
    """
    if isinstance(op0, AbstractCompositeTransform):
        _reject_composed_fibre_offset(op0)
        return None
    if not isinstance(op0, AbstractAdd):
        return None
    # `Boost` is deliberately excluded: its `delta` is a velocity, but its
    # point action is `delta * tau`, NOT the identity — the funnel captures it
    # by differentiation and MUST keep doing so. It falls through today only
    # because it has no `semantic_kind` field; this check makes that
    # intentional, so adding one for symmetry cannot silently reroute `Boost`
    # onto the ladder path and lose its point action.
    from .boost import Boost  # noqa: PLC0415  (boost.py imports AbstractAdd)

    if isinstance(op0, Boost):
        return None
    k = getattr(op0, "semantic_kind", cxr.dpl).order
    return k if k >= 1 else None


def _reject_composed_fibre_offset(op0: AbstractCompositeTransform, /) -> None:
    """Raise if a materialized composite contains a fibre offset.

    The fibre-offset carve-out below can only recognise a fibre offset as the
    *whole* materialized value. A builder that returns ``Translate(shift) |
    Translate(kick, semantic_kind=vel)`` hides one inside a composite, where
    the generic funnel — blind to identity-point-action offsets by
    construction — would silently drop it. Fail loudly instead.
    """
    for child in op0.transforms:
        if _ladder_order(child) is None:
            continue
        raise TypeError(_MSG_COMPOSED_FIBRE_OFFSET.format(cls=type(child).__name__))


def _fibre_offset_order(op: TimeDep, tau: Any, /) -> int | None:
    r"""Ladder order $k$ of ``op``'s materialized fibre offset, else `None`."""
    return _ladder_order(op.materialize(tau))


def _fibre_ladder_op(
    op: TimeDep, op0: AbstractAdd, tau: Any, n: int, rep: Any, /
) -> AbstractAdd:
    r"""Materialize ``op`` with its offset replaced by $d^n\delta/d\tau^n$.

    The $n$-th $\tau$-derivative of a ladder-order-$k$ offset is a tangent
    object of order $k + n = m$, so it is stamped with the data's own
    ``semantic_kind``; the existing static `act` then applies it (including
    the cross-chart Jacobian push) as a matching-order kick.

    ``op0`` is ``op`` already materialized at ``tau`` by the caller (which
    needs it anyway to compute the ladder order) — reused here instead of
    calling ``op.materialize(tau)`` a second time.
    """

    def delta_at(t: Any, /) -> CDict:
        return cast("AbstractAdd", op.materialize(t)).delta

    delta = tau_derivative(delta_at, tau, n=n)
    return replace(op0, delta=delta, semantic_kind=rep.semantic_kind)


@plum.dispatch
def act(
    op: TimeDep,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    geom: cxr.TangentGeometry,
    rep: cxr.Representation,
    /,
    *,
    at: CDict | None = None,
    at_vel: CDict | None = None,
    usys: Any = None,
    **kw: Any,
) -> CDict:
    r"""Tangent action of a `TimeDep`, with the fibre-offset ladder rule.

    A `TimeDep` family whose value is a fibre offset (ladder order
    $k \geq 1$) applies the ladder rule directly: order-$m$ data gains
    $d^{m-k}\delta/d\tau^{m-k}$. Every other `TimeDep` — in particular
    every point-acting one — defers to the generic funnel, which recovers its
    time dependence by differentiating the point action.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.transforms as cxfm

    A velocity kick that grows at 5 km/s2 shifts accelerations by its rate:

    >>> kick = cxfm.TimeDep.from_(lambda t: cxfm.Translate(
    ...     {"x": u.Q(5.0, "km/s2") * t, "y": u.Q(0.0, "km/s"),
    ...      "z": u.Q(0.0, "km/s")},
    ...     chart=cxc.cart3d, semantic_kind=cxr.vel))
    >>> a = {"x": u.Q(1.0, "km/s2"), "y": u.Q(1.0, "km/s2"), "z": u.Q(1.0, "km/s2")}
    >>> out = cxfm.act(kick, u.Q(2.0, "s"), a, cxc.cart3d, cxr.coord_acc)
    >>> out["x"]
    Q(6., 'km / s2')

    """
    m = cast("int", rep.semantic_kind.order)

    # Displacements (m=0) and a missing tau are the funnel's business: it owns
    # the frozen-tau pushforward and the shared tau=None error. Materialize at
    # most once here: `op0` doubles as the routing predicate's input and (on
    # the fibre-offset path) the base for `_fibre_ladder_op`, so it is not
    # re-materialized there.
    op0 = None
    k = None
    if tau is not None and m != 0:
        op0 = op.materialize(tau)
        k = _ladder_order(op0)
    if k is None:
        # Reach the generic funnel itself, not a copy of it. This dispatch is
        # strictly more specific than prolong.py's, so without the explicit
        # invoke every `TimeDep` tangent act would bypass the engine — and
        # any future change there (a new anchor check, a units policy) would
        # silently not apply to the one type that carries all time dependence.
        return cast(
            "CDict",
            _generic_tangent_act()(
                op, tau, x, chart, geom, rep, at=at, at_vel=at_vel, usys=usys, **kw
            ),
        )

    # Lower-order fibres are untouched by a higher-order offset.
    if m < k:
        return x

    op_m = _fibre_ladder_op(op, cast("AbstractAdd", op0), tau, m - k, rep)
    return cast("CDict", cxfmapi.act(op_m, tau, x, chart, rep, at=at, usys=usys, **kw))


@plum.dispatch
def act_jet(
    op: TimeDep,
    tau: Any,
    jet: dict,
    chart: cxc.AbstractChart,
    /,
    *,
    usys: Any = None,
) -> dict:
    """Prolong a `TimeDep`; fibre offsets go slot-wise, the rest generic.

    The jet path needs the same fibre-offset carve-out as ``act`` above: the
    generic jet chain differentiates the point action, which a fibre offset
    does not have.
    """
    if tau is None or _fibre_offset_order(op, tau) is None:
        return prolong_jet(op, tau, jet, chart, usys=usys)
    return _prolong_slotwise(op, tau, jet, chart, usys=usys)


# ============================================================================
# pushforward


@plum.dispatch
def pushforward(
    op: AbstractAdd,
    tau: Any,
    v: CDict,
    chart: cxc.AbstractChart,
    rep: Any,
    /,
    *,
    at: CDict | None = None,
    usys: Any = None,
) -> CDict:
    r"""Pushforward under an additive operator.

    When the offset and the data live in the same Cartesian-type (flat)
    chart — or the operator is a fibre-only offset (ladder order $k \geq 1$,
    identity point action) — the point action's differential is the identity,
    tangent components are unchanged, and no base point is required.

    A $k=0$ offset in a non-flat chart, or acting on data in a different or
    non-flat chart, is not a flat translation: the differential is base-point
    dependent, so this defers to the generic engine, which **requires** the
    base point ``at``.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.transforms as cxfm

    >>> op = cxfm.Translate.from_([1, 2, 3], "km")
    >>> d = {"x": u.Q(1.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}
    >>> cxfm.pushforward(op, None, d, cxc.cart3d, cxr.coord_disp)
    {'x': Q(1., 'km'), 'y': Q(0., 'km'), 'z': Q(0., 'km')}

    """
    # A k=0 offset is a flat translation only when delta and the data live
    # in the same Cartesian-type chart; otherwise the differential is
    # base-point dependent. Defer to the generic engine.
    if not is_componentwise_offset(op, chart):
        return pushforward_generic(op, tau, v, chart, rep, at=at, usys=usys)

    del op, tau, chart, rep, at, usys
    return v


# ============================================================================
# Simplification


@plum.dispatch
def simplify(
    op: AbstractAdd, /, *, approx: bool = True, **kw: Any
) -> AbstractAdd | Identity:
    """Simplify a AbstractAdd operator.

    A translation with zero delta simplifies to Identity. This is a
    value-inspecting rule, so it is skipped when ``approx=False``.

    >>> import coordinax.transforms as cxfm

    >>> op = cxfm.Translate.from_([1, 2, 3], "km")
    >>> cxfm.simplify(op)
    Translate(...)

    >>> op = cxfm.Translate.from_([0, 0, 0], "km")
    >>> cxfm.simplify(op)
    Identity()

    """
    if not approx:
        return op
    is_zero = jtu.all(
        jtu.map(lambda v: jnp.allclose(u.ustrip(AllowValue, v), 0, **kw), op.delta)
    )
    if is_zero:
        return identity
    return op


@plum.dispatch
def _merge(a: AbstractAdd, b: AbstractAdd, /) -> AbstractTransform | None:
    """Merge two adjacent additive operators of the same type and role.

    Offsets of the same operator type and ``semantic_kind`` combine into one
    (their deltas add); anything else is left un-merged.
    """
    if type(a) is not type(b) or getattr(a, "semantic_kind", None) != getattr(
        b, "semantic_kind", None
    ):
        return None
    combined = a + b
    return None if isinstance(combined, Composed) else combined
