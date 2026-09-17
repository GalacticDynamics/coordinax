r"""The tube swept through time: a one-parameter family of tubular slices."""

__all__ = ("SweptTube",)

from collections.abc import Callable
from typing import Any

import equinox as eqx

from .attime import AtTime
from .base import AbstractCurveFrameBuilder, unit_or_none
from .bishop import BishopBuilder
from .chart import TubularChart

_MSG_DIRECTOR_REQUIRED = (
    "`director` is required: `BishopBuilder` transports the normal plane from "
    "a seed, and nothing in the curve fixes that seed. The n-plane gauge is "
    "*director* data -- a Cosserat frame -- and a curve does not carry it: a "
    "rod spinning about its own axis and one at rest trace the same curve, and "
    "have different rates of strain (measured, on a static helix with a spun "
    "director: `K_tau_tau = -0.067526` against exactly `0.0` for a fixed one). "
    "So this cannot be guessed, and guessing it silently reported frame drift "
    "as physical strain (#870). Pass `director=lambda t: <unit 3-vector>`, a "
    "callable of time because a materially spinning rod needs one. Pass "
    "`builder=FrenetSerretBuilder` instead if the frame the curve itself fixes "
    "is what you want -- that one needs no seed and is already equivariant."
)

_MSG_NOT_A_BUILDER = (
    "`builder` must be a builder *class*, not {what}. Every other argument "
    "here is a value, so `builder=BishopBuilder(...)` is the natural slip -- "
    "but this chooses the kind of frame each slice gets and constructs one per "
    "slice itself, from the curve bound at that time. Pass the class: "
    "`builder=BishopBuilder`."
)

_MSG_DIRECTOR_UNUSED = (
    "`director` was given with `builder={name}`, which takes no seed: its "
    "frame is fixed pointwise by the curve, so the director would be silently "
    "ignored. Drop `director`, or pass `builder=BishopBuilder` to use it."
)


class SweptTube(eqx.Module):  # type: ignore[misc]
    r"""One-parameter family of tubular slices: $t \mapsto$ `TubularChart`.

    The 4-D object whose two 3-D sections are the *spatial slice* (time pinned,
    coordinates $(\sigma, n_1, n_2)$) and the *worldtube* (station pinned,
    coordinates $(t, n_1, n_2)$). Callers used to write the family out as a
    lambda; owning it is what lets one gauge be carried across every slice
    rather than each slice picking its own.

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
        on the Bishop path and refused on any other -- see the class notes.
    builder
        Frame builder for each slice. `BishopBuilder` (the default) transports
        from ``director``; `FrenetSerretBuilder` needs no seed.

    Notes
    -----
    ``director`` is a callable of $t$ rather than a fixed vector because a
    materially spinning rod needs a frame that turns with it, and that is a
    physically different tube from one at rest -- which is the whole reason
    the library refuses to supply one.

    It is required **only** on the Bishop path. `FrenetSerretBuilder` fixes
    $\mathbf{N}$ and $\mathbf{B}$ pointwise from the curve, so it is already
    equivariant and a seed would be silently ignored; passing one is a caller
    error rather than a no-op.

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
    builder: type = eqx.field(static=True, kw_only=True, default=BishopBuilder)

    def __check_init__(self) -> None:
        """Require a director on the Bishop path, and refuse one elsewhere.

        A dataclass field cannot be "required iff ``builder`` is Bishop", so
        the rule lives here. `FrenetSerretBuilder` does reject an
        ``initial_normal`` one layer down, but with a different error and only
        once a slice is built -- too late to name the choice that was wrong.
        """
        # Checked before `issubclass`, which raises a bare "arg 1 must be a
        # class" on an instance and -- worse -- accepts any *unrelated* class,
        # sending `builder=dict` into the "takes no seed" branch below to be
        # told it is a seedless builder.
        if not isinstance(self.builder, type):
            what = f"a `{type(self.builder).__name__}` value"
            raise TypeError(_MSG_NOT_A_BUILDER.format(what=what))
        if not issubclass(self.builder, AbstractCurveFrameBuilder):
            raise TypeError(
                _MSG_NOT_A_BUILDER.format(what=f"`{self.builder.__name__}`")
            )

        is_bishop = issubclass(self.builder, BishopBuilder)
        if is_bishop and self.director is None:
            raise ValueError(_MSG_DIRECTOR_REQUIRED)
        if not is_bishop and self.director is not None:
            raise ValueError(_MSG_DIRECTOR_UNUSED.format(name=self.builder.__name__))

    def __call__(self, t: Any, /) -> TubularChart:
        """Return the spatial slice at ``t``."""
        seed = {} if self.director is None else {"initial_normal": self.director(t)}
        return TubularChart(
            self.builder(AtTime(self.curve, t), self.tau_unit, **seed),
            tau_bounds=self.tau_bounds,
        )
