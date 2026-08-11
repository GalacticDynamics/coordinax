"""Equality, hashing, and leaf semantics for parameterized charts."""

from typing import Any

import jax
import jax.numpy as jnp

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm
from coordinax._src.base.charts import AbstractParameterizedChart


class AbstractScaled1D(AbstractParameterizedChart):
    """A minimal parameterized chart -- concrete, despite the name.

    Defined here so the test does not depend on which production chart happens
    to be parameterized. The ``Abstract`` prefix is load-bearing: both
    `coordinax._src.base.charts.is_abstract_class` and the hypothesis package's
    `get_all_subclasses` treat that prefix as "not a real chart", which is the
    only way to keep a throwaway test chart out of the global chart registries
    and the (`functools.cache`-d) hypothesis chart strategies.
    """

    scale: Any  # Quantity (dynamic) or StaticQuantity (static)

    @property
    def components(self) -> tuple[str, ...]:
        return ("x",)

    @property
    def coord_dimensions(self) -> tuple[Any, ...]:
        return (u.dimension("length"),)

    @property
    def cartesian(self) -> "AbstractScaled1D":
        return self


def test_static_parameter_gives_no_leaves_and_hashes() -> None:
    c = AbstractScaled1D(scale=u.StaticQuantity(2.0, "m"))
    assert len(jax.tree.leaves(c)) == 0
    assert hash(c) is not None
    assert c == AbstractScaled1D(scale=u.StaticQuantity(2.0, "m"))
    assert c != AbstractScaled1D(scale=u.StaticQuantity(3.0, "m"))


def test_dynamic_parameter_gives_a_leaf() -> None:
    c = AbstractScaled1D(scale=u.Quantity(2.0, "m"))
    assert len(jax.tree.leaves(c)) == 1


def test_dynamic_charts_compare_conservatively() -> None:
    c = AbstractScaled1D(scale=u.Quantity(2.0, "m"))
    assert c == c  # same object  # noqa: PLR0124
    assert c != AbstractScaled1D(scale=u.Quantity(2.0, "m"))  # not provably equal
    assert c != AbstractScaled1D(scale=u.Quantity(3.0, "m"))


def test_dynamic_chart_is_hashable_on_static_identity() -> None:
    """Hashing must not touch dynamic fields, or dict-keying breaks."""
    c = AbstractScaled1D(scale=u.Quantity(2.0, "m"))
    assert hash(c) is not None
    assert hash(c) == hash(AbstractScaled1D(scale=u.Quantity(99.0, "m")))


def test_equality_is_safe_under_jit() -> None:
    """The case that raises `TracerBoolConversionError` today."""

    @jax.jit
    def f(scale_value: Any) -> Any:
        a = AbstractScaled1D(scale=u.Quantity(scale_value, "m"))
        b = AbstractScaled1D(scale=u.Quantity(scale_value, "m"))
        return jnp.asarray(a == b)  # must not raise

    assert f(2.0) is not None


def test_opaque_non_pytree_field_stays_static() -> None:
    """A field that is not a registered pytree is one *non-array* leaf.

    It must not be treated as a dynamic parameter, or a chart holding such a
    field would silently stop comparing equal to its own twin. Here the field
    is a `StaticQuantity`-radius embedding map: a pytree, but a leafless one.
    """

    def make() -> cxm.EmbeddedChart:
        return cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.StaticQuantity(2.0, "km")))

    assert make() == make()


def test_cross_type_comparison_still_works() -> None:
    assert AbstractScaled1D(scale=u.Quantity(1.0, "m")) != cxc.cart3d
