"""The raw-array path through `pt_map` stays cheap (#719).

`pt_map` on a `dict[str, Quantity]` costs ~20x more *per eager call* than the
same map on bare arrays. Almost none of that is coordinax: of the 178 `plum`
dispatches one such call makes, **two** are `pt_map` itself. The rest are
`unxt`/`quax` resolving arithmetic on `Quantity` operands -- `convert`, `unit`,
`ustrip` -- one per primitive op.

So the cost is not something coordinax can dispatch its way out of; what it can
do is keep the raw-array route open, which is what these tests guard. Counting
dispatches rather than timing keeps the guard deterministic: the numbers below
are exact and repeatable, where wall-clock would flake in CI.
"""

__all__: tuple[str, ...] = ()

import collections

import jax.numpy as jnp
import plum
import pytest

import unxt as u

import coordinax.charts as cxc

_USYS = u.unitsystems.si

#: Ceiling, not a target: raw arrays measured 17 dispatches when written. The
#: assertion is that the fast path has not collapsed into the slow one, so this
#: leaves room to move without becoming a change-detector test.
_RAW_DISPATCH_CEILING = 40


def _count_dispatches(fn):
    """Total `plum` dispatch resolutions performed by ``fn()``."""
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


def test_raw_arrays_cost_far_fewer_dispatches_than_quantities():
    """The gap is the point: ~10x fewer resolutions for the same arithmetic."""
    raw = sum(
        _count_dispatches(
            lambda: cxc.pt_map(_RAW, cxc.sph3d, cxc.cart3d, usys=_USYS)
        ).values()
    )
    qty = sum(
        _count_dispatches(lambda: cxc.pt_map(_QTY, cxc.sph3d, cxc.cart3d)).values()
    )
    assert raw * 4 < qty


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
