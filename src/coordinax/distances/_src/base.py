"""Base classes for distance quantities."""

__all__: tuple[str, ...] = ("AbstractDistance",)


import functools as ft

from typing import Any, TypeVar, cast

import jax
import jax.numpy as jnp
import jax.tree as jt
from plum import add_promotion_rule

import unxt as u

#: Bound to the concrete subclass so `cls._make(...)` types as that
#: subclass. `typing.Self` would read better, but beartype rejects PEP 673
#: hints on methods whose class is not itself `@beartype`-decorated, and this
#: package enables runtime typechecking per-function via an import hook.
_DistT = TypeVar("_DistT", bound="AbstractDistance")


@ft.cache
def _pytree_structure(cls: type["AbstractDistance"], unit: Any, /) -> Any:
    """Return the pytree structure for *cls* at *unit*, with the guard left on.

    The structure carries the static fields -- including ``check_negative`` --
    so unflattening against it yields an instance whose fields, ``repr`` and
    equality are exactly those of a normally-constructed one.

    It is independent of the leaf's shape and dtype (a treedef does not record
    them), so one entry per ``(cls, unit)`` serves every array. The zero-valued
    template is concrete, so building it inside a `jax.jit` trace adds nothing
    to the graph -- verified: the lowered HLO gains no conditional.
    """
    return jt.structure(cls(jnp.zeros(()), unit))


class AbstractDistance(u.AbstractQuantity):
    """Distance quantities."""

    @classmethod
    def _make(  # noqa: PYI019  (see `_DistT`: `Self` breaks beartype here)
        cls: type[_DistT], value: jax.Array, unit: Any, /
    ) -> _DistT:
        """Construct without running ``__check_init__``.

        `Distance` and `Parallax` guard non-negativity with `equinox.error_if`,
        which lowers to a conditional plus two custom-calls on *every*
        construction. This is for the few internal callers whose result cannot
        trip that guard as a matter of arithmetic, so paying for it buys nothing.

        Reconstruction goes through the pytree machinery, which equinox routes
        around ``__init__`` entirely. That makes it cheaper even than passing
        ``check_negative=False``, and unlike that flag it does not leak into the
        ``repr``: the structure carries the static fields, so the result is
        indistinguishable from a normally-constructed instance.

        Reach for this only where the sign is a theorem, and say which one at
        the call site. The current callers rely on ``atan2(1 AU, d)`` lying in
        ``[0, pi]`` for *any* ``d`` -- never negative, because the first
        argument is positive. The interval is closed, not open: ``d = +inf``
        gives exactly ``0`` and ``d = -inf`` gives ``pi``. Zero satisfies the
        guard (it tests ``value < 0``), so the endpoints are harmless here.

        Not every provably-safe site needs it. Where the value is ``10 ** x``,
        XLA already folds the guard away by itself: ``10 ** x`` lowers to
        ``exp(x * ln 10)``, and XLA knows ``exp`` is non-negative, so the
        comparison becomes a constant and the conditional is eliminated. Those
        callers keep the ordinary constructor -- a guard that costs nothing is
        better than a bypass.

        Two things this does *not* do, both consequences of skipping
        ``__init__``:

        - The field ``converter`` is skipped, so *value* must already be an
          array. Passing a list yields an object holding a list.
        - The unit is not validated. *unit* must be one the type accepts.
        """
        return cast(
            "_DistT", jt.unflatten(_pytree_structure(cls, u.unit(unit)), [value])
        )

    @property
    def distance(self) -> "AbstractDistance":  # TODO: more specific type
        """The distance.

        Examples
        --------
        >>> import coordinax.distances as cxd
        >>> d = cxd.Distance(10, "km")
        >>> d.distance is d
        True

        >>> import coordinaxs.astro as cxastro
        >>> cxastro.DistanceModulus(10, "mag").distance
        Distance(1000., 'pc')

        >>> p = cxastro.Parallax(1, "mas")
        >>> p.distance.to("kpc")
        Distance(1., 'kpc')

        """
        from coordinax.distances import Distance  # noqa: PLC0415

        return cast("Distance", Distance.from_(self))


# Add a rule that when a AbstractDistance interacts with a Quantity, the
# distance degrades to a Quantity. This is necessary for many operations, e.g.
# division of a distance by non-dimensionless quantity where the resulting units
# are not those of a distance.
add_promotion_rule(AbstractDistance, u.Q, u.Q)
add_promotion_rule(AbstractDistance, u.quantity.Quantity, u.quantity.Quantity)
add_promotion_rule(AbstractDistance, u.quantity.AbstractAngle, u.Q)
