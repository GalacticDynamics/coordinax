"""Canonicalise the container types of a point's components.

A chart declares what each of its components *is* -- `coord_dimensions` gives
``"angle"`` for an angular coordinate -- but nothing has been enforcing that the
stored value's container agrees. The container instead tracked whichever
arithmetic produced it: `Angle` survives a copy, and degrades to `Quantity`
through `Angle + Quantity` or through any trigonometric call, so the same
coordinate in the same chart came back as `Angle` or `Quantity` depending on the
route taken to reach it.

`Angle` and `Quantity` are distinct pytree nodes, so a route-dependent
container is a route-dependent pytree *structure*: `lax.cond` rejects branches
that disagree, and `jax.tree.map` over two points of the same chart fails.

Dimension belongs in the container; *topology* -- whether a coordinate wraps,
and between which bounds -- belongs in the chart's component domain, not in the
type. So this canonicalises only the container, and reads nothing into it about
branch cuts.
"""

__all__: tuple[str, ...] = ()

from typing import Any

import unxt as u

from coordinaxs.api.custom_types import CDict


def canonical_containers(p: CDict, chart: Any, /) -> CDict:
    """Return `p` with each component in the container its chart declares.

    Angle-dimensioned components become `unxt.Angle`; everything else is passed
    through untouched. Values with no unit (bare arrays) are left alone -- there
    is nothing to re-wrap them as.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> from coordinax._src.charts.containers import canonical_containers

    A plain `Quantity` in an angular slot is promoted, and its unit is kept:

    >>> p = {"r": u.Q(2.0, "m"), "theta": u.Q(90, "deg"), "phi": u.Q(1.0, "rad")}
    >>> canonical_containers(p, cxc.sph3d)
    {'r': Q(2., 'm'), 'theta': Angle(90, 'deg'), 'phi': Angle(1., 'rad')}

    A dimensionless value in an angular slot is left alone rather than being
    forced into an `Angle`, which would raise:

    >>> canonical_containers({"theta": u.Q(1.0, "")}, cxc.sph2)
    {'theta': Q(1., '')}

    When nothing needs promoting the input is returned itself, not a copy:

    >>> p = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
    >>> canonical_containers(p, cxc.cart3d) is p
    True

    """
    # Cheap exit for every chart with no angular component at all -- every
    # Cartesian one, which is the hot path. `pt_map` is already dispatch-bound
    # (#719), so this must not add a dict build per call.
    if "angle" not in chart.coord_dimensions:
        return p

    promoted = {}
    for k, dim in zip(chart.components, chart.coord_dimensions, strict=False):
        v = p.get(k)
        if dim != "angle" or v is None or isinstance(v, u.quantity.AbstractAngle):
            continue
        unit = u.unit_of(v)
        # Promote only what `Angle` will actually accept. A chart may declare a
        # component angular and still be handed a dimensionless value -- a
        # sphere built with a bare `radius=1` does exactly that -- and `Angle`
        # rejects it. Canonicalising a container must never make a value
        # invalid, so anything non-angular is passed through untouched.
        if unit is None or u.dimension_of(unit) != u.dimension("angle"):
            continue
        promoted[k] = u.Angle(u.ustrip(unit, v), unit)

    # Nothing to do is the common case -- every Cartesian chart, and any point
    # already canonical. Returning `p` itself rather than a copy keeps
    # `pt_map(p, chart, chart) is p`, which callers rely on.
    return p if not promoted else {**p, **promoted}
