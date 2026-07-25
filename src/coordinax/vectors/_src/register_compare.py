"""Equivalence comparison for vectors.

``==`` (see `AbstractVector.__eq__`) is *strict*: two vectors are equal only
when they share the same chart, frame, and component data. :func:`equivalent`
is the chart- and unit-*invariant* counterpart -- it asks whether two vectors
denote the *same geometric point*, regardless of the chart used to express it.

It is the vector analogue of the unit-aware "same physical amount" relation on
quantities: where that relaxes a unit-blind ``==`` to compare across units,
``equivalent`` additionally compares across charts.  The dispatch is registered
on the *global* plum ``dispatch`` (the same function `unxt` uses for its own
quantity-level ``equivalent``), *without* importing it from `unxt` -- so a
coordinax vector overload and a unxt quantity overload coexist on one
multiply-dispatched ``equivalent`` when both packages are present, and the
vector overload works standalone otherwise.
"""

__all__: tuple[str, ...] = ("equivalent",)

from typing import Any

import jax.tree as jtu
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import is_any_quantity

import coordinax.representations as cxr
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
    ``atol`` is measured in the Cartesian component units of the first operand
    (or in raw component units for unitless vectors).

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
    # ``equivalent`` is a same-*point* relation, so it is meaningful only for
    # point-geometry vectors (a `Point`, a `Coordinate`).  A `Tangent` denotes a
    # displacement, not a point -- and cannot even be re-expressed in Cartesian
    # without a base point (``to_cartesian`` would raise) -- so any non-point or
    # cross-geometry pair is never "the same point": scalar ``False``, and this
    # guard also keeps the promise that ``equivalent`` never raises.
    if not (
        isinstance(a.rep.geom_kind, cxr.PointGeometry)
        and isinstance(b.rep.geom_kind, cxr.PointGeometry)
    ):
        return jnp.zeros((), dtype=bool)

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

    # Compare component-wise, in the first operand's units.  Promote a unitless
    # (plain-array) leaf to a *dimensionless* quantity so that a unitful-vs-
    # unitless mismatch and an incompatible dimension collapse into one
    # convertibility check (``is_unit_convertible("", "m")`` is ``False``): such
    # components describe different spaces, so the vectors are not equivalent.
    # ``jnp.isclose`` is then unit-aware -- ``atol`` carries the component's unit,
    # so it converts ``b`` into ``a``'s unit itself and never raises here.
    def leaf_close(av: Any, bv: Any) -> Any:
        av = av if is_any_quantity(av) else u.Q(av, "")
        bv = bv if is_any_quantity(bv) else u.Q(bv, "")
        if not u.is_unit_convertible(bv.unit, av.unit):
            return jnp.zeros((), dtype=bool)
        return jnp.isclose(av, bv, rtol=rtol, atol=u.Q(atol, av.unit))

    # ``is_leaf`` stops the tree walk at the `unxt.Quantity` leaves (which are
    # themselves pytrees); ``tree_reduce``'s initializer makes an empty chart (a
    # 0-dimensional Cartesian chart, e.g. ``Cart0D``) vacuously ``True`` -- every
    # point of a 0D space is the same point.  ``isclose`` over unitful leaves
    # yields a dimensionless ``Quantity`` of bools; strip it back to a plain array
    # so the return type matches the scalar-``False`` guards above.
    checks = jtu.map(leaf_close, ac.data, bc.data, is_leaf=is_any_quantity)
    result = jtu.reduce(jnp.logical_and, checks, jnp.ones((), dtype=bool))
    return u.ustrip("", result) if hasattr(result, "unit") else result
