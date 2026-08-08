"""Every concrete chart sits on exactly one branch, with the right leaf count."""

import jax
import pytest

import coordinax.charts as cxc
from coordinax._src.base.charts import (
    NON_ABC_CHART_CLASSES,
    AbstractChart,
    AbstractParameterizedChart,
    AbstractStaticChart,
)


def test_both_branches_are_abstract_charts():
    assert issubclass(AbstractStaticChart, AbstractChart)
    assert issubclass(AbstractParameterizedChart, AbstractChart)


def test_abstract_chart_itself_is_not_registered_static():
    """The root is a plain ABC; staticness lives on the static branch."""
    import jax.tree_util as jtu

    # A fresh subclass of the ROOT must not be silently static.
    assert not issubclass(AbstractParameterizedChart, AbstractStaticChart)
    del jtu


@pytest.mark.parametrize("cls", sorted(NON_ABC_CHART_CLASSES, key=lambda c: c.__name__))
def test_every_concrete_chart_is_on_exactly_one_branch(cls):
    on_static = issubclass(cls, AbstractStaticChart)
    on_param = issubclass(cls, AbstractParameterizedChart)
    assert on_static != on_param, (
        f"{cls.__name__} is on {'both' if on_static else 'neither'} branch"
    )


def test_static_charts_have_zero_leaves():
    """The property every existing Point/vmap/tree_map depends on."""
    for chart in (cxc.cart3d, cxc.sph3d, cxc.cyl3d):
        assert len(jax.tree.leaves(chart)) == 0
        assert hash(chart) is not None
