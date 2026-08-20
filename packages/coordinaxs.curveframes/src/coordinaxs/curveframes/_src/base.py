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
from typing import Any, cast
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

_MSG_TAU_UNIT_DIMENSION = (
    "this curve exposes a parameter of dimension {want}, but `tau_unit` is "
    "{unit!r}, which is {got}. A builder over an arc-length curve needs a "
    "length, e.g. `BishopBuilder(ArcLength(curve, 's'), 'km')`: `ArcLength` "
    "takes the *wrapped* curve's unit, typically a time, while the builder "
    "takes the arc length the wrapper exposes. A common way to land here is "
    "migrating mechanically to 's'. Left as-is, `location` would still return "
    "correct positions, since it never consults the unit, while `tangent` and "
    "`rotation_matrix` would fail later inside the derivative."
)


_MSG_TAU_UNIT_UNINFERABLE = (
    "this builder has no `tau_unit` and was called with a parameter that "
    "carries no unit, so there is nothing to read it off. Either pass a "
    "`Quantity` parameter, e.g. `builder(u.Q(1.0, 's'))`, or declare the unit "
    "on the builder, e.g. `BishopBuilder(curve, 's')`, which is what a raw "
    "(unitless) parameter needs."
)


def unit_or_none(obj: Any, /) -> u.AbstractUnit | None:
    """Field converter: pass `None` through, otherwise coerce to a unit.

    `None` is not "no unit", it is *infer the unit from the parameter the
    builder is called with* -- see `AbstractCurveFrameBuilder._tau_unit_at`.
    """
    return None if obj is None else cast("u.AbstractUnit", u.unit(obj))


def check_param_dimension(curve: Any, tau_unit: u.AbstractUnit, /) -> None:
    """Raise if ``tau_unit`` contradicts the dimension ``curve`` declares.

    Read from the *instance*, not the type, so a wrapper can forward what it
    wraps -- `AtTime(ArcLength(...), t)` still exposes a length, and that is
    the composition the docs recommend, so it is the one most worth catching.

    A curve that declares nothing is unconstrained: most curves are bare
    functions, and a Python callable cannot be asked what unit its argument
    takes.
    """
    want = getattr(curve, "_param_dimension", None)
    if want is None:
        return
    got = str(u.dimension_of(tau_unit))
    if got != want:
        raise ValueError(
            _MSG_TAU_UNIT_DIMENSION.format(want=want, unit=tau_unit, got=got)
        )


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
    tau_unit : AbstractUnit, optional
        Physical unit of the curve parameter (e.g. ``"s"``).  Static: it selects
        the differentiation units, not a numeric value.  `None` (the default)
        reads it off the parameter the builder is called with, which is what
        a `Quantity` already carries -- see `_tau_unit_at`.
    station : Any, optional
        A *fixed* curve parameter — a station along the curve.  When `None`
        (the default) $\tau$ itself is the curve parameter — the classic
        moving-frame usage.  When set, the frame sits at that station and is
        $\tau$-independent: a frame *field* along the curve, differentiable and
        vmappable in ``station``.

    """

    curve: eqx.AbstractVar[Callable[[Any], Any]]
    tau_unit: eqx.AbstractVar[u.AbstractUnit | None]
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

        # A curve that knows what it exposes is checked against a *declared*
        # `tau_unit` here, where the mistake is, rather than left to surface
        # unevenly later: `location` ignores `tau_unit` and returns correct
        # positions, while the autodiff paths raise a conversion error far
        # from the construction that caused it (#718).
        #
        # An undeclared unit has nothing to check yet -- there is no wrong
        # answer to catch until a parameter arrives. `_tau_unit_at` runs the
        # same check on the inferred unit at that point.
        if self.tau_unit is not None:
            check_param_dimension(self.curve, self.tau_unit)

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

    def _tau_unit_at(self, param: Any, /) -> u.AbstractUnit:
        r"""Resolve the curve parameter's unit: declared, or read off ``param``.

        A declared `tau_unit` wins, so the explicit form keeps working
        unchanged. `None` reads `unxt.unit_of` on the parameter actually being
        evaluated -- the station when one is pinned, otherwise the call-time
        $\tau$ -- which is the unit the caller has already stated by handing
        over a `Quantity`.

        Inferring is *safer* than any default, including a correct one: a
        declared unit is a second, independent statement of the same fact, and
        two statements can disagree. `unxt.experimental.jacfwd` converts rather
        than reinterprets, so on a curve that reads its argument with `ustrip`
        the disagreement is absorbed and the answer stays right; on a curve
        that reads `.value` it is not, and the frame comes back silently wrong.
        Reading the unit off the parameter removes the second statement, so
        there is nothing left to disagree.

        Declaring it still earns its place in two cases the parameter cannot
        settle: a curve that reads `.value` rather than converting, and a raw
        (unitless) parameter, which carries no unit to read.
        """
        if self.tau_unit is not None:
            return self.tau_unit
        tau_unit = cast("u.AbstractUnit | None", u.unit_of(param))
        if tau_unit is None:
            raise TypeError(_MSG_TAU_UNIT_UNINFERABLE)
        # Same check `__check_init__` runs on a declared unit -- for an
        # inferred one this is the first moment it can run at all.
        check_param_dimension(self.curve, tau_unit)
        return tau_unit

    def _param(self, tau: Any, /) -> Any:
        r"""Return the curve parameter as a `Quantity`: ``tau``, or the station.

        A raw (unitless) parameter is the **array fastpath**: bare arrays
        carrying no unit, which is the cheapest way in and the reason
        `tau_unit` is still worth declaring. Those numbers mean nothing on
        their own, so they are wrapped here with the declared `tau_unit` --
        the same role a `unxt.AbstractUnitSystem` plays for the raw-array
        route elsewhere in `coordinax`.

        Doing it at this one funnel covers every accessor at once: `__call__`,
        `location`, and both builders' triads all take their curve parameter
        from here, so none of them has to know whether it arrived wrapped.

        A raw parameter with no declared unit is the one combination that
        cannot be served -- nothing anywhere states the unit -- and raises.
        """
        param = tau if self.station is None else self.station
        if u.unit_of(param) is not None:
            return param
        if self.tau_unit is None:
            raise TypeError(_MSG_TAU_UNIT_UNINFERABLE)
        return u.Q(param, self.tau_unit)

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

        >>> builder = cxfc.FrenetSerretBuilder(helix, "s")
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

        >>> cxfc.FrenetSerretBuilder(helix, "s").location(u.Q(0.0, "s"))
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

        >>> cxfc.FrenetSerretBuilder(circle, "s").tangent(u.Q(0.0, "s"))
        Q([-0.,  1.,  0.], '')

        For a two-argument curve ``tau`` is the time -- see `_resolve`.

        """
        b, p = self._resolve(tau)
        R = b.rotation_matrix(p.astype(float))
        return u.Q(R[0], "")
