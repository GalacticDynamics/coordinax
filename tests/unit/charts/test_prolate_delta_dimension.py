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
