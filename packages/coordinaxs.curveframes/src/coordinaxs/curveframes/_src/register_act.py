"""Register ``act`` dispatches for curve-frame transforms.

This module registers two ``plum.dispatch`` overloads of {func}`act` for
`AbstractParallelTransportTransform`:

1. **Quantity path** — ``act(op, tau, x: AbstractQuantity, chart, rep)``
   iterates the internal ``Translate(-gamma) | Rotate(R)`` pipeline, delegating
   to the existing ``act`` dispatches for ``Translate`` and ``Rotate``.
2. **CDict path** — ``act(op, tau, x: CDict, chart, rep)`` does the same for
   component-dictionary inputs.

Because both ``FrenetSerretTransform`` and ``BishopTransform`` inherit from
``AbstractParallelTransportTransform``, a single pair of dispatches covers all
concrete curve-frame transform types.

The ``tau`` (evolution parameter) is threaded through each sub-transform so that
the lazy callables stored on the transform are evaluated at the requested
parameter value.
"""

__all__: tuple[str, ...] = ()

from typing import Any

import plum

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm
import unxt as u

from .base import AbstractParallelTransportTransform
from .custom_types import CDict


@plum.dispatch
def act(
    op: AbstractParallelTransportTransform,
    tau: Any,
    x: u.AbstractQuantity,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> u.AbstractQuantity:
    r"""Apply a parallel-transport transform to a ``Quantity``.

    Evaluates the curve-frame transform at parameter $\tau$ and applies it to
    the input coordinates.  Internally iterates the ``Translate(-\gamma) |
    Rotate(R)`` pipeline, delegating to the existing ``act`` dispatches for
    ``Translate`` and ``Rotate``.

    For a forward transform this computes:

    $$ \mathbf{p}' = R(\tau)\bigl(\mathbf{p}
                  - \boldsymbol{\gamma}(\tau)\bigr)
    $$

    For an inverse transform the fields are already set up so that the same
    pipeline formula gives the inverse mapping.

    Parameters
    ----------
    op : AbstractParallelTransportTransform
        The curve-frame transform (``FrenetSerretTransform`` or
        ``BishopTransform``).
    tau : Quantity
        The evolution parameter at which to evaluate the transform.
    x : AbstractQuantity
        Input 3-D coordinates (e.g. a position vector).
    chart : AbstractChart
        The coordinate chart (typically ``cart3d``).
    rep : Representation
        The representation type (e.g. ``point``).
    **kw
        Additional keyword arguments forwarded to sub-transform ``act``
        dispatches.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.frames as cxf
    >>> import coordinax.transforms as cxfm
    >>> import coordinaxs.curveframes as cxfc

    >>> def circle(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
    ...                           jnp.zeros_like(t)]), "km")

    FrenetSerretTransform:

    >>> fs = cxfc.FrenetSerretTransform.from_curve(circle)
    >>> p = u.Q(jnp.array([1.0, 0.0, 0.0]), "km")
    >>> cxfm.act(fs, u.Q(0.0, "s"), p)
    Q([0., 0., 0.], 'km')

    BishopTransform:

    >>> bt = cxfc.BishopTransform.from_curve(circle)
    >>> result = cxfm.act(bt, u.Q(0.0, "s"), p)
    >>> jnp.allclose(result.value, jnp.array([0., 0., 0.]), atol=1e-5)
    Array(True, dtype=bool)

    """
    # Delegate to Composed, which threads the pipeline AND advances the
    # tangent anchors (at/at_vel) between the time-dependent sub-transforms —
    # a hand-rolled loop that forwards `at` unchanged would evaluate the
    # Rotate's dR/dtau term at the un-translated base point.
    return cxfm.act(cxfm.Composed(op.transforms), tau, x, chart, rep, **kw)  # ty: ignore[invalid-return-type]


@plum.dispatch
def act(
    op: AbstractParallelTransportTransform,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> CDict:
    r"""Apply a parallel-transport transform to a component dictionary.

    Same pipeline as the ``Quantity`` path, but operates on a ``CDict`` (mapping
    of component names to ``Quantity`` values).  Each sub-transform's ``act``
    dispatch handles the ``CDict`` → array → ``CDict`` conversion internally.

    Parameters
    ----------
    op : AbstractParallelTransportTransform
        The curve-frame transform.
    tau : Quantity
        The evolution parameter.
    x : CDict
        Input coordinates as a component dictionary, e.g. ``{"x": Q(1.0, "km"),
        "y": Q(0.0, "km"), "z": Q(0.0, "km")}``.
    chart : AbstractChart
        The coordinate chart.
    rep : Representation
        The representation type.
    **kw
        Additional keyword arguments.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.frames as cxf
    >>> import coordinaxs.curveframes as cxfc

    >>> def circle(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
    ...                           jnp.zeros_like(t)]), "km")

    FrenetSerretTransform:

    >>> fs = cxfc.FrenetSerretTransform.from_curve(circle)
    >>> data = {"x": u.Q(1.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}
    >>> cxfm.act(fs, u.Q(0.0, "s"), data, cx.cart3d, cx.point)
    {'x': Q(0., 'km'), 'y': Q(0., 'km'), 'z': Q(0., 'km')}

    BishopTransform:

    >>> bt = cxfc.BishopTransform.from_curve(circle)
    >>> result = cxfm.act(bt, u.Q(0.0, "s"), data, cx.cart3d, cx.point)
    >>> all(jnp.allclose(result[k].value, 0.0, atol=1e-5) for k in result)
    True

    """
    # Delegate to Composed (see the Quantity overload for why: anchor
    # threading between the time-dependent sub-transforms).
    return cxfm.act(cxfm.Composed(op.transforms), tau, x, chart, rep, **kw)  # ty: ignore[invalid-return-type]


@plum.dispatch
def pushforward(
    op: AbstractParallelTransportTransform,
    tau: Any,
    v: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    *,
    at: CDict | None = None,
    usys: Any = None,
) -> CDict:
    """Frozen-tau pushforward: delegate to the Composed pipeline."""
    return cxfm.pushforward(  # ty: ignore[invalid-return-type]
        cxfm.Composed(op.transforms), tau, v, chart, rep, at=at, usys=usys
    )


@plum.dispatch
def prolong(
    op: AbstractParallelTransportTransform,
    tau: Any,
    jet: dict,
    chart: cxc.AbstractChart,
    /,
    *,
    usys: Any = None,
) -> dict:
    """Kinematic prolongation of a jet: delegate to the Composed pipeline."""
    return cxfm.prolong(  # ty: ignore[invalid-return-type]
        cxfm.Composed(op.transforms), tau, jet, chart, usys=usys
    )
