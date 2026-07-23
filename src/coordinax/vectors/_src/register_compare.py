"""Equivalence comparison for vectors.

``==`` (see `AbstractVector.__eq__`) is *strict*: two vectors are equal only
when they share the same chart, frame, and component data. :func:`equivalent`
is the chart- and unit-*invariant* counterpart -- it asks whether two vectors
denote the *same geometric point*, regardless of the chart used to express it.

This mirrors `unxt.equivalent`, which relaxes ``==`` on quantities from
unit-blind to unit-aware.  The dispatch is registered on the *global* plum
``dispatch`` (not imported from `unxt`), so when `unxt` ships its own
``equivalent`` the two coexist on one multiply-dispatched function.
"""

__all__ = ("equivalent",)

from typing import Any

from plum import dispatch

import quaxed.numpy as jnp

from .base import AbstractVector


@dispatch
def equivalent(
    a: AbstractVector,
    b: AbstractVector,
    /,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Any:
    """Whether two vectors denote the same geometric point.

    Unlike ``==`` -- which is *strict* (equal only for matching chart, frame,
    and data) -- ``equivalent`` is invariant to the chart and to the component
    units: it compares the two vectors as points in a common Cartesian chart.
    It remains *frame-strict*, since coordinates in different frames describe
    different physical points.  Because chart transitions are trigonometric and
    square-root heavy, the comparison is tolerance-based (`rtol`, `atol`);
    ``atol`` is measured in the Cartesian component units of the first operand.

    Examples
    --------
    >>> import coordinax as cx
    >>> import coordinax.charts as cxc

    The same point in Cartesian and spherical charts is *not* ``==`` (the charts
    differ) but *is* ``equivalent``:

    >>> p = cx.Point.from_([1.0, 2.0, 3.0], "m")
    >>> sph = p.cconvert(cxc.sph3d)
    >>> bool(p == sph)
    False
    >>> bool(cx.equivalent(p, sph))
    True

    Equivalence is also invariant to the component units:

    >>> q = cx.Point.from_([1.0, 2.0, 3.0], "km")
    >>> mm = cx.Point.from_([1e6, 2e6, 3e6], "mm")
    >>> bool(cx.equivalent(q, mm))
    True

    Distinct points are not equivalent:

    >>> bool(cx.equivalent(p, cx.Point.from_([1.0, 2.0, 4.0], "m")))
    False

    """
    # Coordinates in different frames describe different physical points.  Chart
    # and frame are static metadata, so this is a plain Python bool -- safe under
    # ``jit`` and mirroring the guard in ``AbstractVector.__eq__``.
    if a.frame != b.frame:
        return jnp.zeros((), dtype=bool)

    # Compare as points in a common Cartesian chart.  This avoids the angle
    # wrapping and coordinate singularities that would make a component-wise
    # comparison in a curvilinear chart unreliable, and keeps the tolerance
    # isotropic in space.
    ac = a.to_cartesian()
    bc = b.to_cartesian()

    # Different Cartesian charts (e.g. a different manifold dimension) can never
    # denote the same point.
    if ac.chart != bc.chart:
        return jnp.zeros((), dtype=bool)

    # Element-wise, per component, expressed in the first operand's units.
    checks = [
        jnp.isclose(
            av.ustrip(av.unit), bc.data[k].ustrip(av.unit), rtol=rtol, atol=atol
        )
        for k, av in ac.data.items()
    ]
    return jnp.all(jnp.stack(jnp.broadcast_arrays(*checks)), axis=0)
