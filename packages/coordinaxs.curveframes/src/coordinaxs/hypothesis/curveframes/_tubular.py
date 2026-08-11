"""Hypothesis strategy for `coordinaxs.curveframes.TubularChart`.

`TubularChart` cannot be derived from its field annotations the way most
chart classes are (see `coordinaxs.hypothesis.charts.chart_init_kwargs`): a
randomly drawn `n_seed` could be 0, and randomly drawn `tau_bounds` could be
degenerate or -- for a closed curve -- span more than one period, which makes
the inverse solve an exact tie (see `TubularChart.tau_bounds`). Instead this
picks a curve from a small fixed set, a builder flavor, and bounds chosen for
that curve.

Importing this module registers the `charts()` and `chart_init_kwargs()`
overloads for `type[TubularChart]` as a side effect (`@plum.dispatch` adds a
method to the shared global `charts`/`chart_init_kwargs` functions). It is
loaded by `coordinaxs.hypothesis.charts` via the `coordinaxs.hypothesis`
entry-point group -- see `coordinaxs.hypothesis.curveframes.__init__` -- so the
overload is always in place before `chart_classes()` can enumerate
`TubularChart`.
"""

__all__ = ("tubular_charts",)

from collections.abc import Callable
from typing import Any

import hypothesis.strategies as st
import jax.numpy as jnp
import plum
from hypothesis import assume

import unxt as u
from coordinaxs.hypothesis.utils import strip_return_annotation

import coordinaxs.curveframes as cxfc


def _circle(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Return a point on the unit circle in the xy-plane. Closed, period 2*pi s."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


def _helix(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Return a point on a helix with pitch along z. Open: no period to respect."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


#: (curve, (lo, hi)) in seconds. `_circle` is closed with period 2*pi, so its
#: bounds span *exactly* one period -- see `TubularChart.tau_bounds`: a wider
#: range makes the inverse solve an exact tie between the two periodic
#: preimages. `_helix` is open, so its bounds just need to be finite. Both
#: curves have nonvanishing curvature everywhere, so both builder flavors
#: (Frenet-Serret included) are well-defined on either -- a straight line is
#: deliberately not offered here, since it is singular for Frenet-Serret.
_CURVES: tuple[
    tuple[Callable[[u.AbstractQuantity], u.AbstractQuantity], tuple[float, float]], ...
] = ((_circle, (0.0, 2 * float(jnp.pi))), (_helix, (-1.0, 6.0)))

_BUILDERS = (cxfc.FrenetSerretBuilder, cxfc.BishopBuilder)


@st.composite
def _tubular_kwargs(draw: st.DrawFn, /) -> dict[str, Any]:
    """Shared draw logic behind both `charts()` and `chart_init_kwargs()`."""
    curve, (lo, hi) = draw(st.sampled_from(_CURVES))
    builder_cls = draw(st.sampled_from(_BUILDERS))
    return {"builder": builder_cls(curve), "tau_bounds": (u.Q(lo, "s"), u.Q(hi, "s"))}


@st.composite
def tubular_charts(draw: st.DrawFn, /) -> cxfc.TubularChart:
    """Strategy for `coordinaxs.curveframes.TubularChart` instances.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinaxs.hypothesis.curveframes as cxfcst

    >>> @given(chart=cxfcst.tubular_charts())
    ... def test_tubular(chart):
    ...     assert chart.components == ("tau", "n1", "n2")

    """
    return cxfc.TubularChart(**draw(_tubular_kwargs()))


@plum.dispatch
@strip_return_annotation
@st.composite
def chart_init_kwargs(
    draw: st.DrawFn,
    chart_class: type[cxfc.TubularChart],
    /,
    *,
    ndim: int | None | st.SearchStrategy = None,
) -> dict[str, Any]:
    """`chart_init_kwargs()` overload for `TubularChart`.

    `tests/unit/charts/test_chart_classes.py`-style tests call
    `chart_init_kwargs(chart_cls)` directly (not only through `charts()`), so
    this needs its own overload rather than relying on the `charts()`
    overload below to shadow the generic annotation-derived one.
    """
    del chart_class, ndim
    return draw(_tubular_kwargs())


@plum.dispatch
@strip_return_annotation
@st.composite
def charts(
    draw: st.DrawFn,
    chart_cls: type[cxfc.TubularChart],
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: int | None | st.SearchStrategy = None,
) -> cxfc.TubularChart:
    """`charts()` overload for `TubularChart`.

    Mirrors the generic `charts(chart_cls: type[cxc.AbstractChart], ...)`
    contract (see `coordinaxs.hypothesis.charts`): `filter`/`exclude` must be
    empty when a concrete class is given, and `ndim` filters after the draw
    rather than constraining it -- `TubularChart` is always 3D, so a mismatched
    `ndim` simply never matches.
    """
    del chart_cls
    if filter or exclude:
        raise ValueError(
            "When chart_cls is provided, filter and exclude must be empty."
        )
    chart = cxfc.TubularChart(**draw(_tubular_kwargs()))
    if ndim is not None:
        assume(chart.ndim == ndim)
    return chart
