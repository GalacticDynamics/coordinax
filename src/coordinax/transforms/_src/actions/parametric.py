"""Parametric (tau-dependent) transform wrapper."""

__all__ = ("Parametric",)

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
    "the parametric transform {op} requires a time parameter; got tau=None."
)


@final
class _FnBuilder(eqx.Module):
    """Wrap a bare ``tau -> AbstractTransform`` function (STATIC: not differentiable).

    Anything the function closes over is a trace-time constant. For
    differentiable parameters, write an `equinox.Module` builder whose
    fields hold them.
    """

    fn: Callable[[Any], AbstractTransform] = eqx.field(static=True)

    def __call__(self, tau: Any, /) -> AbstractTransform:
        return self.fn(tau)


@final
class _ConstBuilder(eqx.Module):
    """A constant family: returns the same transform for every tau."""

    op: AbstractTransform

    def __call__(self, tau: Any, /) -> AbstractTransform:
        del tau
        return self.op


@final
class _ComposedBuilder(eqx.Module):
    """Pointwise-in-tau ``@`` of two families: ``(a @ b)(tau) = a(tau) @ b(tau)``."""

    a: Callable[[Any], AbstractTransform]
    b: Callable[[Any], AbstractTransform]

    def __call__(self, tau: Any, /) -> AbstractTransform:
        return self.a(tau) @ self.b(tau)  # ty: ignore[unsupported-operator]


@final
class _InverseBuilder(eqx.Module):
    """Pointwise inverse of a family: ``binv(tau) = b(tau).inverse``."""

    builder: Callable[[Any], AbstractTransform]

    def __call__(self, tau: Any, /) -> AbstractTransform:
        return self.builder(tau).inverse


@final
class Parametric(AbstractTransform):
    r"""A one-parameter family of transforms: ``builder(tau) -> transform``.

    The builder is typically an `equinox.Module` whose ``__call__(tau)``
    constructs the transform at :math:`\tau`. Its fields (angular
    frequency, phase, curve parameters, ...) are pytree leaves —
    differentiable and vmappable by construction. Extra parameters (e.g.
    an affine curve parameter :math:`\gamma`) are builder fields, not
    call-time arguments.

    Examples
    --------
    >>> import equinox as eqx
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    >>> class RotZ(eqx.Module):
    ...     omega: u.AbstractQuantity
    ...     def __call__(self, tau):
    ...         th = (self.omega * tau).ustrip("rad")
    ...         st, ct = jnp.sin(th), jnp.cos(th)
    ...         R = jnp.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
    ...         return cxfm.Rotate(R)

    >>> op = cxfm.Parametric(RotZ(u.Q(jnp.pi / 2, "rad/s")))
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
    def inverse(self) -> "Parametric":
        """The pointwise inverse of the family: ``inv(tau) = builder(tau).inverse``.

        Examples
        --------
        >>> import equinox as eqx
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.transforms as cxfm

        >>> class RotZ(eqx.Module):
        ...     omega: u.AbstractQuantity
        ...     def __call__(self, tau):
        ...         th = (self.omega * tau).ustrip("rad")
        ...         st, ct = jnp.sin(th), jnp.cos(th)
        ...         R = jnp.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
        ...         return cxfm.Rotate(R)

        >>> op = cxfm.Parametric(RotZ(u.Q(jnp.pi / 2, "rad/s")))
        >>> op.inverse.inverse.builder is op.builder  # involution unwraps
        True

        """
        b = self.builder
        if isinstance(b, _InverseBuilder):  # involution unwraps
            return Parametric(b.builder)
        return Parametric(_InverseBuilder(b))

    @property
    def is_time_dependent(self) -> bool:
        """A `Parametric` transform is, by construction, time-dependent."""
        return True

    def materialize(self, tau: Any, /) -> AbstractTransform:
        """Evaluate the family at ``tau``."""
        if tau is None:
            raise TypeError(_MSG_TAU_REQUIRED.format(op=type(self).__name__))
        return self.builder(tau)

    def _as_builder(self, other: Any, /) -> Callable[[Any], AbstractTransform] | None:
        """Coerce ``other`` to a builder for `@`, or `None` if it cannot."""
        if isinstance(other, Parametric):
            return other.builder
        if isinstance(other, AbstractTransform):
            return _ConstBuilder(other)
        return None

    def __matmul__(self, other: Any, /) -> Any:
        """Pointwise composition: ``(self @ other)(tau) = self(tau) @ other(tau)``.

        Examples
        --------
        >>> import equinox as eqx
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.charts as cxc
        >>> import coordinax.representations as cxr
        >>> import coordinax.transforms as cxfm

        >>> class RotZ(eqx.Module):
        ...     omega: u.AbstractQuantity
        ...     def __call__(self, tau):
        ...         th = (self.omega * tau).ustrip("rad")
        ...         st, ct = jnp.sin(th), jnp.cos(th)
        ...         R = jnp.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
        ...         return cxfm.Rotate(R)

        >>> a = cxfm.Parametric(RotZ(u.Q(0.3, "rad/s")))
        >>> b = cxfm.Parametric(RotZ(u.Q(0.5, "rad/s")))
        >>> ab = a @ b
        >>> isinstance(ab, cxfm.Parametric)
        True

        """
        ob = self._as_builder(other)
        if ob is None:
            return NotImplemented
        return Parametric(_ComposedBuilder(self.builder, ob))

    def __rmatmul__(self, other: Any, /) -> Any:
        """Pointwise composition with `self` applied second."""
        ob = self._as_builder(other)
        if ob is None:
            return NotImplemented
        return Parametric(_ComposedBuilder(ob, self.builder))


# ============================================================================
# Constructors


@Parametric.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Parametric], fn: Callable[[Any], Any], /) -> Parametric:
    """Wrap a bare ``tau -> transform`` function (static, non-differentiable).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    >>> def build(t) -> cxfm.Rotate:
    ...     return cxfm.Rotate(jnp.eye(3))

    >>> op = cxfm.Parametric.from_(build)
    >>> op(u.Q(1.0, "s"), {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")})
    {'x': Q(1., 'm'), 'y': Q(0., 'm'), 'z': Q(0., 'm')}

    """
    return cls(_FnBuilder(fn))


# ============================================================================
# act / pushforward
#
# POINT geometry only: materialize and delegate. Tangent geometry must go
# through the generic tangent funnel (prolong.py) so the d/dtau terms of the
# builder are picked up by the engine's joint (tau, x) jvp — the engine
# differentiates the point action below, and materialization happens inside
# the trace.


@plum.dispatch
def act(
    op: Parametric,
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
    >>> import equinox as eqx
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.transforms as cxfm

    >>> class RotZ(eqx.Module):
    ...     omega: u.AbstractQuantity
    ...     def __call__(self, tau):
    ...         th = (self.omega * tau).ustrip("rad")
    ...         st, ct = jnp.sin(th), jnp.cos(th)
    ...         R = jnp.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
    ...         return cxfm.Rotate(R)

    >>> op = cxfm.Parametric(RotZ(u.Q(jnp.pi / 2, "rad/s")))
    >>> q = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> out = cxfm.act(op, u.Q(1.0, "s"), q, cxc.cart3d, cxr.point)
    >>> out["y"].round(3)
    Q(1., 'm')

    """
    return cast("CDict", cxfmapi.act(op.materialize(tau), tau, x, chart, rep, **kw))


@plum.dispatch
def pushforward(
    op: Parametric,
    tau: Any,
    v: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
) -> CDict:
    """Frozen-tau pushforward: materialize at tau, then push forward."""
    return cast(
        "CDict",
        cxfmapi.pushforward(op.materialize(tau), tau, v, chart, rep, at=at, usys=usys),
    )


# ============================================================================
# Simplification


@plum.dispatch
def simplify(op: Parametric, /, *, approx: bool = True, **kw: Any) -> AbstractTransform:
    """Pass a `Parametric` through unchanged (its value is unknown until tau).

    Examples
    --------
    >>> import equinox as eqx
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    >>> class RotZ(eqx.Module):
    ...     omega: u.AbstractQuantity
    ...     def __call__(self, tau):
    ...         return cxfm.identity

    >>> op = cxfm.Parametric(RotZ(u.Q(1.0, "rad/s")))
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


@plum.dispatch
def _merge(a: Parametric, b: Parametric, /) -> AbstractTransform | None:
    """Merge two adjacent `Parametric` transforms (``a`` applied first)."""
    return _try_matmul(a, b)


@plum.dispatch
def _merge(a: Parametric, b: AbstractTransform, /) -> AbstractTransform | None:
    """Merge a `Parametric` with a constant transform (``a`` applied first)."""
    return _try_matmul(a, b)


@plum.dispatch
def _merge(a: AbstractTransform, b: Parametric, /) -> AbstractTransform | None:
    """Merge a constant transform with a `Parametric` (``a`` applied first)."""
    return _try_matmul(a, b)
