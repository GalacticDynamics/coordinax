"""TimeDep (tau-dependent) transform wrapper."""

__all__ = ("TimeDep",)

from collections.abc import Callable
from typing import Any, cast, final

import equinox as eqx
import plum

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinaxs.api.transforms as cxfmapi
from .base import AbstractTransform
from .custom_types import CDict, OptUSys
from coordinax.transforms._src import groups

_MSG_TAU_REQUIRED = (
    "the time-dependent transform {op} requires a time parameter; got tau=None."
)


@final
class _FnBuilder(eqx.Module):
    """Wrap a bare ``tau -> AbstractTransform`` function in a STATIC field.

    Static refers to what the function *closes over*: those values are
    trace-time constants. The function's ``tau`` dependence is unaffected --
    ``tau`` is a call-time argument, so the prolongation engine differentiates
    it as for any other builder. For differentiable *closed-over* parameters,
    bind them with ``TimeDep.from_(fn, *args)`` (or an explicit
    `equinox.Partial`), or write an `equinox.Module` builder whose fields hold
    them; `TimeDep.from_` leaves either kind unwrapped so their leaves stay
    dynamic.
    """

    fn: Callable[[Any], AbstractTransform] = eqx.field(static=True)

    def __call__(self, tau: Any, /) -> AbstractTransform:
        return self.fn(tau)


@final
class _ConstBuilder(eqx.Module):
    """A constant family: returns the same transform for every tau.

    Only reachable through an EXPLICIT ``TimeDep @ static`` by the caller.
    `simplify` deliberately does not build one — see `_ComposedBuilder`.
    """

    op: AbstractTransform

    def __call__(self, tau: Any, /) -> AbstractTransform:
        del tau
        return self.op


@final
class _ComposedBuilder(eqx.Module):
    """Pointwise-in-tau composition: ``(a @ b)(tau) = a(tau) @ b(tau)``.

    Only some transforms (e.g. `Rotate`) implement ``@``; everything else
    composes with ``|``, which has the same order semantics.

    That ``|`` fallback is why `simplify` never builds one of these: for a
    fibre offset it yields, at every tau, a `Composed` holding an offset of
    ladder order >= 1 -- exactly the spelling `add.py` rejects. Only an
    EXPLICIT ``a @ b`` gets here, where the caller has asked for pointwise
    composition and owns that constraint.
    """

    a: Callable[[Any], AbstractTransform]
    b: Callable[[Any], AbstractTransform]

    def __call__(self, tau: Any, /) -> AbstractTransform:
        a, b = self.a(tau), self.b(tau)
        out = _try_matmul(a, b)
        return a | b if out is None else out


@final
class _InverseBuilder(eqx.Module):
    """Pointwise inverse of a family: ``binv(tau) = b(tau).inverse``."""

    builder: Callable[[Any], AbstractTransform]

    def __call__(self, tau: Any, /) -> AbstractTransform:
        return self.builder(tau).inverse


@final
class TimeDep(AbstractTransform):
    r"""A one-parameter family of transforms: ``builder(tau) -> transform``.

    The builder is typically an `equinox.Module` whose ``__call__(tau)``
    constructs the transform at :math:`\tau`. Its fields (angular
    frequency, phase, curve parameters, ...) are pytree leaves —
    differentiable and vmappable by construction. Extra parameters (e.g.
    an affine curve parameter :math:`\gamma`) are builder fields, not
    call-time arguments. A user-defined function works just as well --
    see `TimeDep.from_`; :math:`\tau` is a call-time argument either way,
    so its derivatives always flow into `act` and `act_jet`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    >>> zhat = jnp.array([0.0, 0.0, 1.0])
    >>> op = cxfm.TimeDep(cxfm.RotationAboutAxis(u.Q(90, "deg/s"), axis=zhat))
    >>> q = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> out = op(u.Q(1.0, "s"), q)
    >>> out["y"].round(3)
    Q(1., 'm')

    """

    builder: Callable[[Any], AbstractTransform]

    @classmethod
    def groups(cls) -> frozenset[type]:
        """Return the groups to which this map belongs."""
        del cls
        return frozenset((groups.DiffeomorphismGroup,))

    @property
    def inverse(self) -> "TimeDep":
        """The pointwise inverse family, ``inv(tau) = builder(tau).inverse``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.transforms as cxfm

        >>> zhat = jnp.array([0.0, 0.0, 1.0])
        >>> op = cxfm.TimeDep(cxfm.RotationAboutAxis(u.Q(90, "deg/s"), axis=zhat))
        >>> op.inverse.inverse.builder is op.builder  # involution unwraps
        True

        """
        b = self.builder
        if isinstance(b, _InverseBuilder):  # involution unwraps
            return TimeDep(b.builder)
        return TimeDep(_InverseBuilder(b))

    @property
    def is_time_dependent(self) -> bool:
        """A `TimeDep` transform is, by construction, time-dependent."""
        return True

    def evaluate_at(self, tau: Any, /) -> AbstractTransform:
        """Evaluate the family at ``tau``."""
        if tau is None:
            raise TypeError(_MSG_TAU_REQUIRED.format(op=type(self).__name__))
        return self.builder(tau)

    def _as_builder(self, other: Any, /) -> Callable[[Any], AbstractTransform] | None:
        """Coerce ``other`` to a builder for `@`, or `None` if it cannot."""
        if isinstance(other, TimeDep):
            return other.builder
        if isinstance(other, AbstractTransform):
            return _ConstBuilder(other)
        return None

    def __matmul__(self, other: Any, /) -> Any:
        """Pointwise composition: ``(self @ other)(tau) = self(tau) @ other(tau)``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.transforms as cxfm

        >>> zhat = jnp.array([0.0, 0.0, 1.0])
        >>> a = cxfm.TimeDep(cxfm.RotationAboutAxis(u.Q(0.3, "rad/s"), axis=zhat))
        >>> b = cxfm.TimeDep(cxfm.RotationAboutAxis(u.Q(0.5, "rad/s"), axis=zhat))
        >>> ab = a @ b
        >>> isinstance(ab, cxfm.TimeDep)
        True

        """
        ob = self._as_builder(other)
        if ob is None:
            return NotImplemented
        return TimeDep(_ComposedBuilder(self.builder, ob))

    def __rmatmul__(self, other: Any, /) -> Any:
        """Pointwise composition with `self` applied second."""
        ob = self._as_builder(other)
        if ob is None:
            return NotImplemented
        return TimeDep(_ComposedBuilder(ob, self.builder))


# ============================================================================
# Constructors


@TimeDep.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[TimeDep], fn: Callable[..., Any], /, *args: Any, **kw: Any
) -> TimeDep:
    """Build from a ``tau -> transform`` callable.

    ``tau`` is a call-time argument, never a stored parameter, so a
    user-defined function's ``tau`` dependence is fully differentiated by the
    kinematic-prolongation engine -- even for a bare lambda. `act` on tangent
    data and `act_jet` pick up the resulting ``d/dtau`` terms automatically.

    Storage matters only for the builder's *other* parameters. With no extra
    arguments: a bare function/lambda cannot be a pytree leaf, so it is stored
    STATIC -- anything it *closes over* is a trace-time constant and a fresh
    closure forces a `jit` retrace. A callable that is already a pytree (an
    `equinox.Module`, notably `equinox.Partial`) is used as the builder as-is,
    keeping its leaves dynamic -- differentiable and `jit`-cached. (Wrapping
    it would set those leaves static, silently destroying them.)

    Given extra ``*args``/``**kw``, they are bound onto ``fn`` with
    `equinox.Partial`, which keeps them as dynamic leaves. **``fn`` must take
    ``tau`` LAST** -- ``fn(param..., tau, **kw)`` -- because `equinox.Partial`
    prepends its bound positionals: ``partial(tau)`` calls
    ``fn(*bound, tau, **kw)``. Writing ``fn(tau, param)`` instead silently
    passes the parameter as tau. The resulting builder is a `equinox.Partial`,
    so the operator needs `equinox.filter_jit`, not plain `jax.jit`.

    Examples
    --------
    >>> import equinox as eqx
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxfm

    A bare lambda is enough for the ``tau`` derivatives to flow: this drift at
    3 km/s, acted on data at rest, transforms the velocity to exactly the
    drift rate.

    >>> rate = {"x": u.Q(3.0, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")}
    >>> drift = cxfm.TimeDep.from_(
    ...     lambda t: cxfm.Translate({k: v * t for k, v in rate.items()},
    ...                              chart=cxc.cart3d)
    ... )

    >>> at_rest = cx.Coordinate(
    ...     point=cx.Point.from_([0.0, 0.0, 0.0], "km"),
    ...     velocity=cx.Tangent.from_([0.0, 0.0, 0.0], "km/s"),
    ... )
    >>> cxfm.act(drift, u.Q(0.0, "s"), at_rest)["velocity"]["x"]
    Q(3., 'km / s')

    Extra arguments are bound as leaves -- keeping them differentiable and
    `jit`-cached. Note ``t`` comes last:

    >>> def scaled(factor, t) -> cxfm.Scale:
    ...     return cxfm.Scale.from_factors(jnp.full(3, factor))

    >>> op = cxfm.TimeDep.from_(scaled, jnp.asarray(2.0))
    >>> [float(x) for x in jax.tree.leaves(eqx.filter(op, eqx.is_array))]
    [2.0]

    A pre-built `equinox.Partial` is equivalent:

    >>> op = cxfm.TimeDep.from_(eqx.Partial(scaled, jnp.asarray(2.0)))
    >>> [float(x) for x in jax.tree.leaves(eqx.filter(op, eqx.is_array))]
    [2.0]

    """
    if args or kw:
        return cls(eqx.Partial(fn, *args, **kw))
    return cls(fn if isinstance(fn, eqx.Module) else _FnBuilder(fn))


# ============================================================================
# act / pushforward
#
# POINT geometry only: evaluate and delegate. Tangent geometry must go
# through the generic tangent funnel (prolong.py) so the d/dtau terms of the
# builder are picked up by the engine's joint (tau, x) jvp — the engine
# differentiates the point action below, and evaluation happens inside
# the trace.


@plum.dispatch
def act(
    op: TimeDep,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    geom: cxr.PointGeometry,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> CDict:
    """Point action: evaluate the family at tau, then act.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.transforms as cxfm

    >>> zhat = jnp.array([0.0, 0.0, 1.0])
    >>> op = cxfm.TimeDep(cxfm.RotationAboutAxis(u.Q(90, "deg/s"), axis=zhat))
    >>> q = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> out = cxfm.act(op, u.Q(1.0, "s"), q, cxc.cart3d, cxr.point)
    >>> out["y"].round(3)
    Q(1., 'm')

    """
    return cast("CDict", cxfmapi.act(op.evaluate_at(tau), tau, x, chart, rep, **kw))


@plum.dispatch
def pushforward(
    op: TimeDep,
    tau: Any,
    v: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
) -> CDict:
    r"""Frozen-$\tau$ pushforward: evaluate at tau, then push forward.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.transforms as cxfm

    >>> zhat = jnp.array([0.0, 0.0, 1.0])
    >>> op = cxfm.TimeDep(cxfm.RotationAboutAxis(u.Q(90, "deg/s"), axis=zhat))
    >>> v = {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")}
    >>> out = cxfm.pushforward(op, u.Q(1.0, "s"), v, cxc.cart3d, cxr.coord_vel)
    >>> out["y"].round(3)
    Q(1., 'm / s')

    """
    return cast(
        "CDict",
        cxfmapi.pushforward(op.evaluate_at(tau), tau, v, chart, rep, at=at, usys=usys),
    )


# ============================================================================
# Simplification


@plum.dispatch
def simplify(op: TimeDep, /, *, approx: bool = True, **kw: Any) -> AbstractTransform:
    """Pass a `TimeDep` through unchanged (its value is unknown until tau).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    >>> zhat = jnp.array([0.0, 0.0, 1.0])
    >>> op = cxfm.TimeDep(cxfm.RotationAboutAxis(u.Q(1.0, "rad/s"), axis=zhat))
    >>> cxfm.simplify(op) is op
    True

    """
    del approx, kw
    return op


def _try_matmul(a: Any, b: Any, /) -> AbstractTransform | None:
    """Compute ``a @ b``, or return `None` if that combination is unsupported."""
    try:
        out = a @ b
    except TypeError:
        return None
    return None if out is NotImplemented else out


# NOTE: there is deliberately no `_merge` rule for `TimeDep`. Folding two
# families into one requires the `_ComposedBuilder` `|` fallback, which for a
# fibre offset materializes a `Composed` that `add.py` rejects -- turning a
# working pipeline into one that raises. `Composed` already represents the
# pair correctly; the merge was only ever an optimization. An explicit
# `a @ b` still composes pointwise: that is the caller asking for it.
