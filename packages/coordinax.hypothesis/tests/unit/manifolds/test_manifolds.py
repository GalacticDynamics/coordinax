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


@given(M_cls=cxst.manifold_classes())
def test_manifold_classes_returns_concrete_manifold_subclasses(
    M_cls: type[cxm.AbstractManifold],
) -> None:
    """manifold_classes returns concrete manifold subclasses."""
    assert issubclass(M_cls, cxm.AbstractManifold)
    assert M_cls is not cxm.AbstractManifold


@given(M=cxst.manifolds())
def test_manifolds_generates_valid_manifold_instances(M: cxm.AbstractManifold) -> None:
    """manifolds generates valid manifold instances."""
    assert isinstance(M, cxm.AbstractManifold)
    assert isinstance(M.atlas, cxm.AbstractAtlas)
    assert M.atlas.ndim == M.ndim
    assert isinstance(M.default_chart(), cxc.AbstractChart)


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


@given(M=cxst.manifolds(cxm.CustomManifold))
def test_custom_manifold_strategy_basic(M: cxm.CustomManifold) -> None:
    """manifolds(CustomManifold) generates valid CustomManifold objects."""
    # Strategy output should always be the concrete manifold wrapper.
    assert isinstance(M, cxm.CustomManifold)
    # The inherited manifold contract guarantees default_chart is usable.
    assert M.has_chart(M.default_chart())
    # Manifold dimension is forwarded from atlas dimension.
    assert M.default_chart().ndim == M.ndim


@given(
    M=cxst.manifolds(
        cxm.CustomManifold, ndim=2, required_chart_classes=(cxc.Cart2D, cxc.Polar2D)
    )
)
def test_custom_manifold_required_chart_classes(M: cxm.CustomManifold) -> None:
    """required_chart_classes are forwarded for CustomManifold draws."""
    assert M.has_chart(cxc.cart2d)
    assert M.has_chart(cxc.polar2d)


@given(M=cxst.manifolds(st.just(cxm.CustomManifold)))
def test_custom_manifold_from_strategy_selector(M: cxm.CustomManifold) -> None:
    """SearchStrategy manifold_cls draws then redispatches to typed generation."""
    assert isinstance(M, cxm.CustomManifold)
    assert M.has_chart(M.default_chart())


@given(
    atlas=cxst.atlases(
        cxm.CustomAtlas, ndim=2, required_chart_classes=(cxc.Cart2D, cxc.Polar2D)
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


@given(M=st.from_type(cxm.CustomManifold))
def test_custom_manifold_from_type_registration(M: cxm.CustomManifold) -> None:
    """st.from_type(CustomManifold) resolves to the registered strategy."""
    assert isinstance(M, cxm.CustomManifold)
    assert M.has_chart(M.default_chart())
