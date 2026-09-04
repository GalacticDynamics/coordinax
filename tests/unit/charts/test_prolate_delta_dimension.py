"""`ProlateSpheroidal3D.Delta` is a focal length, and only a length."""

import jax
import pytest

import unxt as u

import coordinax.charts as cxc


@pytest.mark.parametrize("unit", ["m", "kpc", "AU"])
@pytest.mark.parametrize("container", [u.Q, u.quantity.StaticQuantity])
def test_a_length_is_accepted_in_either_container(container, unit) -> None:
    """Both quantity types stay available: differentiability is opt-in."""
    chart = cxc.ProlateSpheroidal3D(Delta=container(2.0, unit))
    assert u.dimension_of(chart.Delta) == u.dimension("length")


@pytest.mark.parametrize("unit", ["s", "kpc2", "", "km/s"])
def test_a_non_length_is_rejected(unit) -> None:
    """The hole this closes.

    `check_data` compares `mu` against ``Delta**2``, so seconds gave a bound
    in ``s2`` and the chart then validated ``s2`` data while declaring its
    components ``area``. Rejecting at construction is what makes the declared
    dimensions mean anything.
    """
    with pytest.raises(ValueError, match="must have dimensions of length"):
        cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, unit))


def test_the_error_names_the_dimension_it_got() -> None:
    """A bare "invalid Delta" would not say which of the two rules was broken."""
    with pytest.raises(ValueError, match=r"got s \(time\)"):
        cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "s"))


def test_a_dynamic_delta_still_traces() -> None:
    """The check reads the unit, which is static, so it survives `jit`.

    A check that touched `Delta.value` would not: that is a tracer under
    `jit`, and the chart is built inside traced code by every `pt_map` that
    takes one as an argument.
    """
    build = jax.jit(lambda d: cxc.ProlateSpheroidal3D(Delta=d).Delta)
    assert u.dimension_of(build(u.Q(2.0, "kpc"))) == u.dimension("length")


def test_the_check_stays_off_the_hot_paths() -> None:
    """`__post_init__` must not re-run when the chart is rebuilt from its pytree.

    This is what keeps the check a per-construction cost rather than a
    per-call one: `jit` flattens and unflattens a chart argument on every
    boundary crossing, and equinox rebuilds a `Module` without calling
    `__init__`. If that ever changed, the cost would move onto the traced
    path, where it does not belong -- and nothing else would go red.
    """
    chart = cxc.ProlateSpheroidal3D(Delta=u.Q(2.0, "kpc"))
    leaves, treedef = jax.tree.flatten(chart)

    calls = 0
    original = type(chart).__post_init__

    def counting(self: object) -> None:
        nonlocal calls
        calls += 1
        original(self)

    type(chart).__post_init__ = counting  # type: ignore[method-assign]
    try:
        jax.tree.unflatten(treedef, leaves)
        jax.jit(lambda c: c.Delta)(chart)
    finally:
        type(chart).__post_init__ = original  # type: ignore[method-assign]

    assert calls == 0


def test_the_unit_check_is_memoised() -> None:
    """The check is answered once per unit, not once per construction.

    `unxt.is_unit_convertible` walks astropy's unit graph and costs about
    twice what building the whole chart does. Uncached it dominated
    construction; the cache is what makes the check affordable at all.
    """
    from coordinax._src.charts.d3 import _is_length

    _is_length.cache_clear()
    unit = u.unit("kpc")
    _is_length(unit)
    before = _is_length.cache_info().misses
    for _ in range(100):
        _is_length(unit)
    assert _is_length.cache_info().misses == before
    assert _is_length.cache_info().hits >= 100
