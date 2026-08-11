"""Every concrete chart sits on exactly one branch, with the right leaf count."""

import jax
import pytest

import unxt as u

import coordinax.charts as cxc
from coordinax._src.base.charts import (
    NON_ABC_CHART_CLASSES,
    AbstractParameterizedChart,
    AbstractStaticChart,
)


@pytest.mark.parametrize("cls", sorted(NON_ABC_CHART_CLASSES, key=lambda c: c.__name__))
def test_every_concrete_chart_is_on_exactly_one_branch(cls):
    on_static = issubclass(cls, AbstractStaticChart)
    on_param = issubclass(cls, AbstractParameterizedChart)
    assert on_static != on_param, (
        f"{cls.__name__} is on {'both' if on_static else 'neither'} branch"
    )
    # Branch membership is not the same as being registered. A static chart that
    # inherits the branch but never reaches `jtu.register_static` -- e.g. an
    # intermediate base overriding `__init_subclass__` without calling super() --
    # silently becomes one opaque leaf. Check the property, not the ancestry.
    if on_static:
        assert not jax.tree.leaves(cls.__new__(cls))


# ---------------------------------------------------------------------------
# A static chart must not be handed an array-bearing parameter; see
# `AbstractStaticChart.__post_init__`.


def _prolate(delta):
    return cxc.ProlateSpheroidal3D(Delta=delta)


def test_product_chart_rejects_a_parameterized_factor():
    with pytest.raises(TypeError, match="static chart"):
        cxc.CartesianProductChart(
            factors=(_prolate(u.Q(2.0, "m")), cxc.cart3d), factor_names=("a", "b")
        )


def test_galilean_ct_rejects_a_parameterized_spatial_chart():
    with pytest.raises(TypeError, match="static chart"):
        cxc.GalileanCT(spatial_chart=_prolate(u.Q(2.0, "m")))


def test_a_static_delta_factor_is_still_accepted():
    """The guard must not cost the legitimate case: a `StaticQuantity` factor."""
    factor = _prolate(u.StaticQuantity(2.0, "m"))
    product = cxc.CartesianProductChart(
        factors=(factor, cxc.cart3d), factor_names=("a", "b")
    )
    assert len(jax.tree.leaves(product)) == 0
    assert hash(product) is not None
    assert len(jax.tree.leaves(cxc.GalileanCT(spatial_chart=factor))) == 0


def test_a_tracer_cannot_escape_through_a_static_chart():
    """The harm the guard exists to stop, at the point where it happens."""

    @jax.jit
    def f(delta):
        return cxc.GalileanCT(spatial_chart=_prolate(delta))

    with pytest.raises(TypeError, match="static chart"):
        f(u.Q(2.0, "m"))
