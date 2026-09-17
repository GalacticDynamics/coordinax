r"""The tube swept through time: a one-parameter family of tubular slices."""

__all__ = ("SweptTube",)

from collections.abc import Callable
from typing import Any

import equinox as eqx

from .arclength import _is_two_argument
from .attime import AtTime
from .base import AbstractCurveFrameBuilder, unit_or_none
from .bishop import BishopBuilder
from .chart import TubularChart

_MSG_ONE_ARGUMENT_CURVE = (
    "`SweptTube` needs a two-argument `gamma(tau, t)`: it binds the time per "
    "slice with `AtTime`, and a one-argument curve has no slot for it. A tube "
    "that does not change with time is still spelled with both, "
    "`lambda tau, t: gamma(tau)`, which says so rather than leaving it to be "
    "inferred from an arity."
)


_MSG_DIRECTOR_REQUIRED = (
    "`director` is required: `{name}` fixes its normal plane from `{arg}`, and "
    "nothing in the curve fixes that -- a rod spinning about its own axis and "
    "one at rest trace the same curve and strain differently, so it cannot be "
    "guessed (#870). Pass `director=lambda t: <unit 3-vector>`, or "
    "`builder=FrenetSerretBuilder` for the frame the curve itself fixes."
)

_MSG_NOT_A_BUILDER = (
    "`builder` must be a builder *class*, not {what}: `SweptTube` constructs "
    "one per slice itself, from the curve bound at that time. Pass the class, "
    "`builder=BishopBuilder`, not an instance."
)

_MSG_DIRECTOR_UNUSED = (
    "`director` was given with `builder={name}`, which takes no seed: its "
    "frame is fixed pointwise by the curve, so the director would be silently "
    "ignored. Drop `director`, or pass `builder=BishopBuilder` to use it."
)


def _check_builder(builder: Any, /) -> None:
    """Require ``builder`` to be a builder *class*, before `issubclass` sees it.

    `issubclass` raises a bare "arg 1 must be a class" on an instance, and
    accepts any *unrelated* class. Module-level so it can be tested directly:
    with runtime typechecking on, the ``builder: type`` annotation rejects a
    bad value first, so the raises below fire only with it off.
    """
    if not isinstance(builder, type):
        raise TypeError(
            _MSG_NOT_A_BUILDER.format(what=f"a `{type(builder).__name__}` value")
        )
    if not issubclass(builder, AbstractCurveFrameBuilder):
        raise TypeError(_MSG_NOT_A_BUILDER.format(what=f"`{builder.__name__}`"))


class SweptTube(eqx.Module):  # type: ignore[misc]
    r"""One-parameter family of tubular slices: $t \mapsto$ `TubularChart`.

    The 4-D object whose two 3-D sections are the *spatial slice* (time pinned,
    coordinates $(\tau, n_1, n_2)$ -- the names `TubularChart` actually uses)
    and the *worldtube* (station pinned, coordinates $(t, n_1, n_2)$).
    Owning the family is what lets one gauge be settled once, not per slice.

    Parameters
    ----------
    curve
        The two-argument $\gamma(\tau, t)$.
    tau_unit
        Unit of the curve parameter.
    tau_bounds
        Scan range for each slice's inverse solve.
    director
        The n-plane gauge, as a callable of $t$ returning a 3-vector. Required
        exactly when ``builder.gauge_field`` names an argument, and refused
        when it does not -- see the class notes.
    builder
        Frame builder for each slice. `BishopBuilder` (the default) transports
        from ``director``; `FrenetSerretBuilder` needs no seed.

    Notes
    -----
    ``director`` is a callable of $t$ rather than a fixed vector because a
    materially spinning rod needs a frame that turns with it, and that is a
    physically different tube from one at rest -- which is the whole reason
    the library refuses to supply one.

    Whether it is required is the *builder's* statement, read from
    `AbstractCurveFrameBuilder.gauge_field`: ``"normal_0"`` for
    `BishopBuilder`'s transport seed, ``"plane_normal"`` for
    `SignedPlanarBuilder`'s plane, and `None` for `FrenetSerretBuilder`, which
    fixes $\mathbf{N}$ and $\mathbf{B}$ pointwise from the curve and would
    silently ignore one.

    Examples
    --------
    >>> import jax.numpy as jnp, unxt as u
    >>> import coordinaxs.curveframes as cxfc

    >>> def rod(tau, t):
    ...     s = tau.ustrip("km"); tv = t.ustrip("s")
    ...     return u.Q(jnp.stack([s * (1 + 0.5 * tv), 0.1 * tv * s**2,
    ...                           jnp.zeros_like(s)]), "km")

    >>> tube = cxfc.SweptTube(rod, "km",
    ...                       tau_bounds=(u.Q(0.0, "km"), u.Q(3.0, "km")),
    ...                       director=lambda t: jnp.asarray([0.0, 0.0, 1.0]))
    >>> chart = tube(u.Q(1.0, "s"))
    >>> chart.components
    ('tau', 'n1', 'n2')

    """

    curve: Any
    tau_unit: Any = eqx.field(static=True, converter=unit_or_none)
    tau_bounds: tuple[Any, Any] = eqx.field(kw_only=True)
    director: Callable[[Any], Any] | None = eqx.field(kw_only=True, default=None)
    builder: type[AbstractCurveFrameBuilder] = eqx.field(
        static=True, kw_only=True, default=BishopBuilder
    )

    def __check_init__(self) -> None:
        """Pair ``director`` with the builder's gauge: required, or refused.

        A dataclass field cannot be "required iff the builder has a gauge", and
        the layer below only refuses once a slice is built -- too late to name
        the choice that was wrong.
        """
        _check_builder(self.builder)

        if not _is_two_argument(self.curve):
            raise ValueError(_MSG_ONE_ARGUMENT_CURVE)

        gauge = self.builder.gauge_field
        if gauge is not None and self.director is None:
            raise ValueError(
                _MSG_DIRECTOR_REQUIRED.format(name=self.builder.__name__, arg=gauge)
            )
        if gauge is None and self.director is not None:
            raise ValueError(_MSG_DIRECTOR_UNUSED.format(name=self.builder.__name__))

    def __call__(self, t: Any, /) -> TubularChart:
        """Return the spatial slice at ``t``."""
        # `__check_init__` has paired these: a gauge name implies a director.
        gauge = self.builder.gauge_field
        seed = {gauge: self.director(t)} if gauge and self.director else {}
        return TubularChart(
            self.builder(AtTime(self.curve, t), self.tau_unit, **seed),
            tau_bounds=self.tau_bounds,
        )
