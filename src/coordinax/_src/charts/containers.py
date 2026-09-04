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

#: Resolved once. `u.dimension("angle")` is ~4us, and this runs per component.
_ANGLE = u.dimension("angle")

#: `u.dimension_of` is ~68us per call and `pt_map` is already dispatch-bound
#: (#719), so the answer is memoised on the unit. Units are few, hashable and
#: immutable, so the cache is bounded by the units a program actually uses.
_IS_ANGULAR: dict[object, bool] = {}


def _angular(unit: object, /) -> bool:
    """Return whether `unit` has angular dimension, memoised."""
    try:
        return _IS_ANGULAR[unit]
    except KeyError:
        result = u.dimension_of(unit) == _ANGLE
        _IS_ANGULAR[unit] = result
        return result


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
        # Promote only what `Angle` will accept. A chart may declare a
        # component angular and still be handed a dimensionless value -- a
        # sphere built with a bare `radius=1` does exactly that -- and `Angle`
        # rejects it. Canonicalising a container must never make a value
        # invalid, so anything non-angular is passed through untouched.
        #
        # `v.unit` rather than `u.unit_of(v)`: on an `AbstractQuantity` the
        # field is exactly what `unit_of` returns, and reading it costs ~0.5us
        # against ~8us of dispatch. Anything without one has no unit to
        # promote from.
        if not isinstance(v, u.AbstractQuantity) or not _angular(v.unit):
            continue
        unit = v.unit
        # `_mk` writes the fields and returns, skipping the `plum`-dispatched
        # field converters and `__check_init__`. Its precondition is an
        # already-normalised value and unit, which holds here: both come off
        # `v`, an `AbstractQuantity` that normalised them on its own
        # construction, and `_angular` above has just established the angular
        # dimension that `AbstractAngle.__check_init__` would re-derive.
        #
        # unxt#904 memoised that check, so it is no longer what `_mk` is
        # avoiding -- the `plum`-dispatched field converters are, and #904 did
        # not touch those. Both costs are live here: #904 landed in unxt 2.0.4,
        # which the lock resolves, while the declared floor is 2.0.2 and
        # `test_oldest` builds against it. Measured on either side of #904:
        #
        #                                 pre-#904     2.0.4
        #   u.Angle(value, unit)          170.9us -> 28.9us
        #   u.Angle._mk(...)                0.8us ->  0.8us
        #   canonical_containers, checked 351.8us -> 66.8us
        #   canonical_containers, `_mk`     8.7us ->  8.5us
        #
        # Canonicalising a spherical point still costs ~8x more through the
        # checked constructor even on 2.0.4 (66.8us against 8.5us), so this
        # stays. That is the ratio the choice turns on: the constructors
        # themselves differ by ~36x, but only the angular components of one
        # point go through them. In an eager `pt_map` the whole saving is ~2%
        # of the call, near the noise -- if the converters ever get the same
        # treatment, drop `_mk` and this paragraph with it.
        promoted[k] = u.Angle._mk(value=v.value, unit=unit)

    # Nothing to do is the common case -- every Cartesian chart, and any point
    # already canonical. Returning `p` itself rather than a copy keeps
    # `pt_map(p, chart, chart) is p`, which callers rely on.
    return p if not promoted else {**p, **promoted}
