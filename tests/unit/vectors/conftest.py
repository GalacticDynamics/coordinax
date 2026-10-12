"""Shared helpers for the vector tests."""

import unxt as u

import coordinax.charts as cxc


def jit_constructible(chart, /):
    """Whether a vector on ``chart`` can be built under `jax.jit` at all.

    Not a `ProlateSpheroidal3D` whose ``Delta`` is a `u.StaticQuantity` (what
    `cxst.charts()` draws): its `check_data` `eqx.error_if`s on that ``Delta``,
    which has no array to thread the error onto, so even the raw constructor
    raises ``No arrays to thread error on to`` under `jit`. A `u.Q` ``Delta`` is
    fine. Pre-existing on main and unrelated to the dimension check.
    """
    return not (
        isinstance(chart, cxc.ProlateSpheroidal3D)
        and isinstance(chart.Delta, u.StaticQuantity)
    )


def wrong_dimension(data, chart, dims, /):
    """Swap the first unit-carrying component for one of another dimension.

    Returns ``(key, data)``, or ``(None, None)`` if no component is checkable: a
    ``None`` dimension is unconstrained and a bare array has no unit.
    """
    for k, d in zip(chart.components, dims, strict=True):
        if d is not None and isinstance(data[k], u.AbstractQuantity):
            unit = "m" if d == u.dimension("time") else "s"
            return k, {**data, k: u.Q(1.0, unit)}
    return None, None
