r"""Bind the evaluation time of a time-dependent curve."""

__all__ = ("AtTime",)

from collections.abc import Callable
from typing import Any, final

import equinox as eqx


@final
class AtTime(eqx.Module):
    r"""Bind the evaluation time of a two-argument curve.

    A time-dependent curve $\boldsymbol{\gamma}(\tau, t)$ takes a curve
    parameter $\tau$ and an evaluation time $t$. `AtTime` freezes $t$ and
    returns the one-argument curve
    $\tilde{\boldsymbol{\gamma}}(\tau) = \boldsymbol{\gamma}(\tau, t)$, so it
    can be used anywhere a one-argument curve is expected (e.g. wrapped in a
    `BishopBuilder`, or a `curve` for `ArcLength`).

    See `ArcLength`'s docstring for why binding time *before* wrapping in
    `ArcLength` (``ArcLength(AtTime(curve, t))``) means something different
    from binding it *after* (``AtTime(ArcLength(curve), t)``): the former
    freezes arc length to a fixed slice, the latter measures arc length on
    whatever slice is evaluated.

    Parameters
    ----------
    curve : Callable
        A function ``(tau, t) -> Quantity[float, (3,)]``. Split at
        construction (`equinox.partition` on `equinox.is_array`) into its own
        array leaves -- kept dynamic, so a differentiable ``curve`` (e.g. an
        `ArcLength`) stays differentiable through `AtTime` -- and everything
        else, which is static. A bare function has no array leaves of its
        own, so it contributes nothing to `AtTime`'s pytree; only ``t`` does.
    t
        The evaluation time to bind. Pass a `unxt.Quantity` to make it a
        differentiable pytree leaf, or a `unxt.StaticQuantity` to keep
        `AtTime` leaf-free.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinaxs.curveframes as cxfc

    >>> def stretch(tau: u.Q, t: u.Q) -> u.Q:
    ...     x = tau.ustrip("s") * (1.0 + 0.5 * t.ustrip("s"))
    ...     z = jnp.zeros_like(x)
    ...     return u.Q(jnp.stack([x, z, z]), "km")

    >>> curve = cxfc.AtTime(stretch, u.Q(1.0, "s"))
    >>> curve(u.Q(2.0, "s"))
    Q([3., 0., 0.], 'km')

    """

    _curve_dynamic: Any
    _curve_static: Any = eqx.field(static=True)

    t: Any
    """The bound evaluation time."""

    def __init__(self, curve: Callable[[Any, Any], Any], t: Any, /) -> None:
        self._curve_dynamic, self._curve_static = eqx.partition(curve, eqx.is_array)
        self.t = t

    @property
    def curve(self) -> Callable[[Any, Any], Any]:
        """The wrapped, time-dependent curve, reassembled from its two halves."""
        return eqx.combine(self._curve_dynamic, self._curve_static)

    def __call__(self, tau: Any, /) -> Any:
        """Evaluate the wrapped curve at the bound time."""
        return self.curve(tau, self.t)
