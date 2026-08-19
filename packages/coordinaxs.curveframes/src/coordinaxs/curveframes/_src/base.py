r"""Abstract base classes for curve-attached reference frames.

This module defines the two abstract base classes on which the entire
``coordinaxs.curveframes`` package is built:

* `AbstractParallelTransportFrame` — a curve-attached orthonormal reference
  frame that integrates with the ``coordinax.frames`` frame-transition system.
* `AbstractCurveFrameBuilder` — the `equinox.Module` builder that a
  `coordinax.transforms.TimeDep` wraps: ``builder(tau)`` returns the
  rigid-body transform ``Translate(-gamma) | Rotate(R)`` at that parameter.

"""

__all__ = ("AbstractCurveFrameBuilder", "AbstractParallelTransportFrame")

import abc
import dataclasses

from collections.abc import Callable
from jaxtyping import Array
from typing import Any
from typing_extensions import TypeVar

import equinox as eqx

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.transforms as cxfm
import unxt as u

from .arclength import _is_two_argument
from .attime import AtTime

FrameT = TypeVar(
    "FrameT", bound=cxf.AbstractReferenceFrame, default=cxf.AbstractReferenceFrame
)

#: `_resolve` returns a builder of the *same* concrete type it was called on,
#: which is what lets a subclass reach its own helpers on the result.
#:
#: `Self` says this more directly, and ruff's PYI019 rewrites to it -- but
#: `beartype` raises `BeartypeDecorHintPep673Exception` on PEP 673 in a method
#: whose class it does not itself decorate, so `Self` fails at *runtime* here.
#: Hence the explicit TypeVar, and the `noqa` at the signature.
BuilderT = TypeVar("BuilderT", bound="AbstractCurveFrameBuilder")

_MSG_TWO_ARGUMENT_NEEDS_STATION = (
    "this curve takes two positional arguments, `gamma(tau, t)`, so the "
    "builder's call-time parameter is the time `t` and the station along the "
    "curve must be pinned with `station=`. Without it both are unbound and no "
    "transform can be built. Either pass `station=<value>` to get a frame at "
    "a fixed station that evolves with `t`, or bind the slice instead with "
    "`AtTime(curve, t)`, which makes the curve one-argument so the call-time "
    "parameter is the curve parameter again."
)


class AbstractParallelTransportFrame(cxf.AbstractTransformedReferenceFrame[FrameT]):
    r"""Abstract base class for curve-attached orthonormal frames in 3-D.

    A parallel-transport frame attaches an orthonormal triad to each point of a
    smooth space curve $\boldsymbol{\gamma}(\tau)$.  Two concrete flavours are
    provided:

    * `FrenetSerretFrame` — axes are (T, N, B) from the Frenet--Serret
      apparatus; singular when curvature vanishes.
    * `BishopFrame` — axes are (T, U1, U2) obtained by parallel transport;
      well-defined for all regular curves.

    Inherits from `coordinax.frames.AbstractTransformedReferenceFrame` and
    therefore carries three fields:

    - ``base_frame`` — the ambient reference frame (e.g. ``Alice()``).
    - ``xop`` — the forward transform (base → curve frame), a
      `coordinax.transforms.TimeDep` wrapping an
      `AbstractCurveFrameBuilder`.
    - ``xop_inv`` — pre-computed inverse of ``xop`` (curve frame → base).

    Because this class is a  ``AbstractTransformedReferenceFrame``, the generic
    ``frame_transition`` dispatches apply automatically.  Concrete subclasses
    must be ``@final``.

    Notes
    -----
    The evolution parameter $\tau$ is **not** stored on the frame object.  It is
    supplied at evaluation time when the frame-transition operator is applied to
    coordinates via ``act(op, tau, x)``.

    Examples
    --------
    Concrete subclasses are used directly; see `FrenetSerretFrame` and
    `BishopFrame` for usage examples.

    """


# ============================================================================


class AbstractCurveFrameBuilder(eqx.Module):
    r"""ABC for curve-frame builders: ``tau -> Translate(-gamma) | Rotate(R)``.

    At each parameter value $\tau$, the built transform maps an ambient point
    $\mathbf{p}$ to curve-frame coordinates:

    $$
        \mathbf{p}' = R(\tau)\bigl(\mathbf{p}
                      - \boldsymbol{\gamma}(\tau)\bigr)
    $$

    decomposed into ``Translate(-gamma) | Rotate(R)``.

    Being an `equinox.Module`, every field is a pytree leaf: the curve's own
    parameters (when ``curve`` is itself an `equinox.Module`) and ``station``
    are differentiable and vmappable.  A bare function passed as ``curve`` still
    works, but whatever it closes over is a trace-time constant.

    Fields
    ------
    curve : Callable
        The curve $\tau \mapsto \boldsymbol{\gamma}(\tau)$, mapping a
        parameter `Quantity` to a Cartesian 3-vector `Quantity`.
    tau_unit : AbstractUnit
        Physical unit of the curve parameter (e.g. ``"s"``).  Static: it selects
        the differentiation units, not a numeric value.
    station : Any, optional
        A *fixed* curve parameter — a station along the curve.  When `None`
        (the default) $\tau$ itself is the curve parameter — the classic
        moving-frame usage.  When set, the frame sits at that station and is
        $\tau$-independent: a frame *field* along the curve, differentiable and
        vmappable in ``station``.

    """

    curve: eqx.AbstractVar[Callable[[Any], Any]]
    tau_unit: eqx.AbstractVar[u.AbstractUnit]
    station: eqx.AbstractVar[Any]

    def __check_init__(self) -> None:
        """Require a station when the curve is two-argument.

        `equinox` runs this for every concrete builder, so `BishopBuilder`
        and `FrenetSerretBuilder` are both covered here rather than each
        repeating the check.

        A builder is called with a *single* parameter. For a one-argument
        curve that parameter is the curve parameter. For a two-argument
        ``gamma(tau, t)`` it is the time, which leaves the station to be
        supplied as a field -- and with neither pinned, two unknowns face one
        slot and no transform can be produced. That is arithmetic, not
        policy, so it is refused at construction rather than at every call.

        What such a caller usually means is already spelled
        ``AtTime(curve, t)``: that binds the slice, making the curve
        one-argument again, so the call-time parameter goes back to being
        the curve parameter. The message names it.

        An uninspectable curve raises out of `_is_two_argument`, matching
        how `ArcLength` treats one.
        """
        if _is_two_argument(self.curve) and self.station is None:
            raise ValueError(_MSG_TWO_ARGUMENT_NEEDS_STATION)

    def _resolve(  # noqa: PYI019  (see BuilderT: `Self` breaks beartype)
        self: BuilderT, tau: Any, /
    ) -> tuple[BuilderT, Any]:
        r"""Reduce to a one-argument builder and the parameter to evaluate it at.

        For an ordinary one-argument curve this is the identity: the builder
        is already evaluable and ``tau`` is already the curve parameter.

        For a two-argument curve, ``tau`` is the *time*, and the frame is the
        one belonging to that time slice, taken at the pinned station. A
        slice of $\gamma(\tau, t)$ at fixed $t$ is a one-argument curve --
        exactly what `AtTime` produces -- so the whole of the existing
        machinery applies to it unchanged, parallel-transport ODE included.
        Nothing about the frame mathematics is time-dependent; only which
        curve it runs on.

        Returning the builder rather than mutating keeps this usable from
        `__call__`, `location` and `tangent` alike, and keeps the two-argument
        path a routing decision made in exactly one place.
        """
        if _is_two_argument(self.curve):
            return dataclasses.replace(
                self, curve=AtTime(self.curve, tau)
            ), self.station
        return self, tau

    # ---------------------------------------------------------------

    def _param(self, tau: Any, /) -> Any:
        """Return the curve parameter: ``tau``, or the fixed ``station``."""
        return tau if self.station is None else self.station

    @abc.abstractmethod
    def rotation_matrix(self, tau: Any, /) -> Array:
        r"""Evaluate the $3 \times 3$ rotation matrix $R$ at ``tau``.

        The rows of $R$ are the frame vectors, e.g.
        $[\mathbf{T};\,\mathbf{N};\,\mathbf{B}]$ for Frenet--Serret.
        """
        raise NotImplementedError  # pragma: no cover

    def __call__(self, tau: Any, /) -> cxfm.Composed:
        r"""Build the transform at ``tau``: ``Translate(-gamma) | Rotate(R)``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.transforms as cxfm
        >>> import coordinaxs.curveframes as cxfc

        >>> def helix(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), t]), "m")

        >>> builder = cxfc.FrenetSerretBuilder(helix)
        >>> op = builder(u.Q(0.0, "s"))
        >>> isinstance(op, cxfm.Composed)
        True

        For a two-argument curve ``tau`` is the time, and the result is the
        rigid motion of that time slice at the pinned station -- see
        `_resolve`.

        """
        b, p = self._resolve(tau)
        cart = cxc.cart3d
        g = b._param(p)
        translate = cxfm.Translate(cxc.cdict(-b.curve(g), cart), chart=cart)
        return translate | cxfm.Rotate(b.rotation_matrix(p))

    # ---------------------------------------------------------------
    # Convenience accessors

    def location(self, tau: Any, /) -> Any:
        r"""Evaluate the curve position $\boldsymbol{\gamma}$ at ``tau``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def helix(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), t]), "m")

        >>> cxfc.FrenetSerretBuilder(helix).location(u.Q(0.0, "s"))
        Q([1., 0., 0.], 'm')

        For a two-argument curve ``tau`` is the time -- see `_resolve`.

        """
        b, p = self._resolve(tau)
        return b.curve(b._param(p))

    def tangent(self, tau: Any, /) -> u.Q:
        r"""Return the unit tangent vector $\mathbf{T}$ (row 0 of R).

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> cxfc.FrenetSerretBuilder(circle).tangent(u.Q(0.0, "s"))
        Q([-0.,  1.,  0.], '')

        For a two-argument curve ``tau`` is the time -- see `_resolve`.

        """
        b, p = self._resolve(tau)
        R = b.rotation_matrix(p.astype(float))
        return u.Q(R[0], "")
