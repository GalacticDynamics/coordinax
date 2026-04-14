"""Tests for manifold strategies."""

import hypothesis.strategies as st
from hypothesis import given

import coordinax.charts as cxc
import coordinax.manifolds as cxm

import coordinax.hypothesis.main as cxst


@given(atlas_cls=cxst.atlas_classes())
def test_atlas_classes_returns_concrete_atlas_subclasses(
    atlas_cls: type[cxm.AbstractAtlas],
) -> None:
    """atlas_classes returns concrete atlas subclasses."""
    assert issubclass(atlas_cls, cxm.AbstractAtlas)
    assert atlas_cls is not cxm.AbstractAtlas


@given(atlas=cxst.atlases())
def test_atlases_generates_valid_atlas_instances(atlas: cxm.AbstractAtlas) -> None:
    """atlases generates valid atlas instances."""
    assert isinstance(atlas, cxm.AbstractAtlas)
    assert isinstance(atlas.default_chart(), cxc.AbstractChart)
    assert atlas.default_chart() in atlas


@given(manifold_cls=cxst.manifold_classes())
def test_manifold_classes_returns_concrete_manifold_subclasses(
    manifold_cls: type[cxm.AbstractManifold],
) -> None:
    """manifold_classes returns concrete manifold subclasses."""
    assert issubclass(manifold_cls, cxm.AbstractManifold)
    assert manifold_cls is not cxm.AbstractManifold


@given(manifold=cxst.manifolds())
def test_manifolds_generates_valid_manifold_instances(
    manifold: cxm.AbstractManifold,
) -> None:
    """manifolds generates valid manifold instances."""
    assert isinstance(manifold, cxm.AbstractManifold)
    assert isinstance(manifold.atlas, cxm.AbstractAtlas)
    assert manifold.atlas.ndim == manifold.ndim
    assert isinstance(manifold.default_chart, cxc.AbstractChart)


@given(atlas=cxst.atlases(cxm.CustomAtlas))
def test_custom_atlas_strategy_basic(atlas: cxm.CustomAtlas) -> None:
    """atlases(CustomAtlas) generates valid CustomAtlas objects."""
    # Strategy output should always be the concrete type we requested.
    assert isinstance(atlas, cxm.CustomAtlas)
    # Atlas stores chart registrations as an ordered tuple.
    assert isinstance(atlas.charts, tuple)
    assert len(set(atlas.charts)) == len(atlas.charts)
    # The default chart is required to be drawn from the registered class set.
    assert isinstance(atlas.default_chart(), tuple(atlas.charts))
    # Core atlas invariant: the default chart must be supported.
    assert atlas.has_chart(atlas.default_chart())


@given(manifold=cxst.manifolds(cxm.CustomManifold))
def test_custom_manifold_strategy_basic(manifold: cxm.CustomManifold) -> None:
    """manifolds(CustomManifold) generates valid CustomManifold objects."""
    # Strategy output should always be the concrete manifold wrapper.
    assert isinstance(manifold, cxm.CustomManifold)
    # The inherited manifold contract guarantees default_chart is usable.
    assert manifold.has_chart(manifold.default_chart)
    # Manifold dimension is forwarded from atlas dimension.
    assert manifold.default_chart.ndim == manifold.ndim


@given(
    manifold=cxst.manifolds(
        cxm.CustomManifold,
        ndim=2,
        required_chart_classes=(cxc.Cart2D, cxc.Polar2D),
    )
)
def test_custom_manifold_required_chart_classes(manifold: cxm.CustomManifold) -> None:
    """required_chart_classes are forwarded for CustomManifold draws."""
    assert manifold.has_chart(cxc.cart2d)
    assert manifold.has_chart(cxc.polar2d)


@given(manifold=cxst.manifolds(st.just(cxm.CustomManifold)))
def test_custom_manifold_from_strategy_selector(manifold: cxm.CustomManifold) -> None:
    """SearchStrategy manifold_cls draws then redispatches to typed generation."""
    assert isinstance(manifold, cxm.CustomManifold)
    assert manifold.has_chart(manifold.default_chart)


@given(
    atlas=cxst.atlases(
        cxm.CustomAtlas,
        ndim=2,
        required_chart_classes=(cxc.Cart2D, cxc.Polar2D),
    )
)
def test_required_chart_classes_are_present(atlas: cxm.CustomAtlas) -> None:
    """required_chart_classes are always included for CustomAtlas draws."""
    # Required classes were requested explicitly in strategy parameters.
    assert cxc.Cart2D in atlas.charts
    assert cxc.Polar2D in atlas.charts
    # Membership must hold for canonical instances of those classes.
    assert atlas.has_chart(cxc.cart2d)
    assert atlas.has_chart(cxc.polar2d)


@given(atlas=cxst.atlases(st.just(cxm.CustomAtlas)))
def test_custom_atlas_from_strategy_selector(atlas: cxm.CustomAtlas) -> None:
    """SearchStrategy atlas_cls draws then redispatches to typed generation."""
    assert isinstance(atlas, cxm.CustomAtlas)
    assert atlas.has_chart(atlas.default_chart())


@given(atlas=st.from_type(cxm.CustomAtlas))
def test_custom_atlas_from_type_registration(atlas: cxm.CustomAtlas) -> None:
    """st.from_type(CustomAtlas) resolves to the registered strategy."""
    assert isinstance(atlas, cxm.CustomAtlas)
    assert atlas.has_chart(atlas.default_chart())


@given(manifold=st.from_type(cxm.CustomManifold))
def test_custom_manifold_from_type_registration(manifold: cxm.CustomManifold) -> None:
    """st.from_type(CustomManifold) resolves to the registered strategy."""
    assert isinstance(manifold, cxm.CustomManifold)
    assert manifold.has_chart(manifold.default_chart)
