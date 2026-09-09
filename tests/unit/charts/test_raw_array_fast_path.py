"""Neither route through `pt_map` is dispatch-heavy (#719).

`pt_map` on a `dict[str, Quantity]` used to cost ~20x more *per eager call*
than the same map on bare arrays, and this file was written on the reading
that coordinax could not dispatch its way out of it: of the 178 `plum`
dispatches one such call made, only **two** were `pt_map` itself, the rest
being `unxt`/`quax` resolving arithmetic on `Quantity` operands.

That reading was wrong about the cause. The dispatches were per *primitive
operation*, and a body only performs them because it does its arithmetic on
`Quantity` operands -- which is the body's choice, not a property of units.
Bodies that resolve their units once through `strip`/`wrap` and compute on raw
arrays make far fewer: `sph3d -> cart3d` went from 178 dispatches to 45, and
from ~2890us to ~700us, for identical output.

So the guard has changed shape. It used to assert a *gap* between the two
routes; closing that gap is now the goal, and what is worth pinning is that
neither route is expensive. The raw route must stay open and cheap (it moved
17 -> 23 dispatches, the `unit_of` probes `strip` makes even on bare values,
which cost nothing measurable), and the quantity route must not drift back
toward per-primitive dispatch.

Counting dispatches rather than timing keeps the guard deterministic: the
numbers are exact and repeatable, where wall-clock would flake in CI. Counting
also survives a change in how `plum` caches -- `plum#290` would let the
unfaithful signatures here be cached, cutting the *cost* of a dispatch without
changing how many happen.
"""

__all__: tuple[str, ...] = ()

import collections

import jax.numpy as jnp
import plum
import pytest

import unxt as u

import coordinax.charts as cxc

_USYS = u.unitsystems.si

#: Ceiling, not a target: raw arrays measured 17 dispatches when written and 23
#: once `strip` began probing `unit_of` on them. The assertion is that the fast
#: path has not collapsed into the slow one, so this leaves room to move without
#: becoming a change-detector test.
_RAW_DISPATCH_CEILING = 40

#: Likewise for quantities: 178 before the body stopped computing on `Quantity`
#: operands, 45 after. A body reverted to per-primitive `Quantity` arithmetic
#: lands back near 178, which this catches well before then.
_QTY_DISPATCH_CEILING = 90


def _count_dispatches(fn):
    """Count calls to `plum`-dispatched functions made by ``fn()``.

    Calls, not resolutions: this patches `plum.Function.__call__`, so a cache
    hit is counted like any other call. That is the number the guard wants --
    it does not move when `plum` changes what it caches -- but it is not a
    count of distinct signature resolutions. See the module docstring.
    """
    counts = collections.Counter()
    original = plum.Function.__call__

    def counting(self, *args, **kwargs):
        counts[self.__name__] += 1
        return original(self, *args, **kwargs)

    plum.Function.__call__ = counting
    try:
        fn()
    finally:
        plum.Function.__call__ = original
    return counts


_RAW = {"r": jnp.asarray(1.0), "theta": jnp.asarray(0.5), "phi": jnp.asarray(0.3)}
_QTY = {"r": u.Q(1.0, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0.3, "rad")}


def test_the_raw_array_route_agrees_with_the_quantity_route():
    """The fast path is only worth having if it is the same map."""
    out_q = cxc.pt_map(_QTY, cxc.sph3d, cxc.cart3d)
    out_a = cxc.pt_map(_RAW, cxc.sph3d, cxc.cart3d, usys=_USYS)
    for k in ("x", "y", "z"):
        assert float(u.ustrip("m", out_q[k])) == float(out_a[k])


def test_raw_arrays_stay_off_the_unit_machinery():
    counts = _count_dispatches(
        lambda: cxc.pt_map(_RAW, cxc.sph3d, cxc.cart3d, usys=_USYS)
    )
    assert sum(counts.values()) <= _RAW_DISPATCH_CEILING


def test_quantities_stay_off_the_per_primitive_dispatch_path():
    """The body resolves its units once, not once per arithmetic primitive.

    This is the guard that replaced an assertion that the two routes differ by
    ~10x. They no longer do, and that is the improvement rather than a
    regression -- see the module docstring.
    """
    counts = _count_dispatches(lambda: cxc.pt_map(_QTY, cxc.sph3d, cxc.cart3d))
    assert sum(counts.values()) <= _QTY_DISPATCH_CEILING


@pytest.mark.parametrize(
    ("from_chart", "to_chart", "point"),
    [
        (cxc.sph3d, cxc.cart3d, _RAW),
        (
            cxc.cart3d,
            cxc.sph3d,
            {k: jnp.asarray(v) for k, v in (("x", 1.0), ("y", 0.5), ("z", 0.3))},
        ),
        (
            cxc.cyl3d,
            cxc.cart3d,
            {k: jnp.asarray(v) for k, v in (("rho", 1.0), ("phi", 0.5), ("z", 0.3))},
        ),
        (
            cxc.polar2d,
            cxc.cart2d,
            {k: jnp.asarray(v) for k, v in (("r", 1.0), ("theta", 0.5))},
        ),
    ],
    ids=["sph3d->cart3d", "cart3d->sph3d", "cyl3d->cart3d", "polar2d->cart2d"],
)
def test_the_raw_route_is_open_for_every_common_pair(from_chart, to_chart, point):
    """A gap here would silently push a pure-JAX pipeline onto the slow path."""
    out = cxc.pt_map(point, from_chart, to_chart, usys=_USYS)
    assert not any(isinstance(v, u.AbstractQuantity) for v in out.values())
