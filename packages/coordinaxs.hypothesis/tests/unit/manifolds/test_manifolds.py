"""Tests for manifold strategies."""

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from hypothesis.errors import Unsatisfiable

import coordinax.charts as cxc
import coordinax.manifolds as cxm

import coordinaxs.hypothesis.main as cxst
from coordinaxs.hypothesis.manifolds._src.atlas import _atlas_class_supports_ndim
from coordinaxs.hypothesis.manifolds._src.manifold import _manifold_class_supports_ndim
from coordinaxs.hypothesis.utils import get_all_subclasses


@given(atlas_cls=cxst.atlas_classes())
def test_atlas_classes_returns_concrete_atlas_subclasses(
    atlas_cls: type[cxm.AbstractAtlas],
) -> None:
    """atlas_classes returns concrete atlas subclasses."""
    assert issubclass(atlas_cls, cxm.AbstractAtlas)
    assert atlas_cls is not cxm.AbstractAtlas


@given(atlas=cxst.atlases())
def test_atlases_generates_valid_atlas_instances(atlas: cxm.AbstractAtlas) -> None:
    """Atlases generates valid atlas instances."""
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
    """Manifolds generates valid manifold instances."""
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


class TestNdimIsHonoured:
    """``ndim=`` pins the dimensionality; it is never silently clamped."""

    @pytest.mark.parametrize("ndim", [0, 1, 2, 3])
    def test_euclidean_atlas_matches_requested_ndim(self, ndim: int) -> None:
        """A supported ``ndim`` yields an atlas of exactly that dimensionality."""

        @given(atlas=cxst.atlases(cxm.EuclideanAtlas, ndim=ndim))
        @settings(max_examples=10, deadline=None)
        def check(atlas: cxm.EuclideanAtlas) -> None:
            assert atlas.ndim == ndim

        check()

    @pytest.mark.parametrize("ndim", [-1, 4, 5])
    def test_euclidean_atlas_discards_unsupported_ndim(self, ndim: int) -> None:
        """An unsupported ``ndim`` is discarded, not clamped into range.

        ``max(0, min(target_ndim, 3))`` used to hand back a 3-D atlas for
        ``ndim=5`` and a 0-D one for ``ndim=-1``, quietly violating the
        documented contract.
        """

        @given(atlas=cxst.atlases(cxm.EuclideanAtlas, ndim=ndim))
        @settings(max_examples=10, deadline=None)
        def check(atlas: cxm.EuclideanAtlas) -> None:
            pytest.fail(f"ndim={ndim} should be unsatisfiable, got {atlas!r}")

        with pytest.raises(Unsatisfiable):
            check()

    @pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5, 6])
    def test_product_manifold_factor_dims_sum_to_ndim(self, ndim: int) -> None:
        """Factor dimensionalities partition the requested total exactly."""

        @given(M=cxst.manifolds(cxm.CartesianProductManifold, ndim=ndim))
        @settings(max_examples=20, deadline=None)
        def check(M: cxm.CartesianProductManifold) -> None:
            assert sum(f.ndim for f in M.factors) == ndim
            assert all(f.ndim >= 1 for f in M.factors)
            assert M.ndim == ndim

        check()

    @pytest.mark.parametrize("ndim", [1, 2, 3, 4])
    def test_product_atlas_factor_dims_sum_to_ndim(self, ndim: int) -> None:
        """Same contract for the atlas-level product partition."""

        @given(atlas=cxst.atlases(cxm.CartesianProductAtlas, ndim=ndim))
        @settings(max_examples=20, deadline=None)
        def check(atlas: cxm.CartesianProductAtlas) -> None:
            assert sum(f.ndim for f in atlas.factors) == ndim

        check()


def _assert_drawable(strategy: st.SearchStrategy, cls: type) -> None:
    """Draw a few examples from *strategy*, expecting instances of *cls*.

    A separate function rather than an inline closure: `given` rejects default
    arguments, and closing over a loop variable trips ``ruff``'s B023.
    """

    @given(obj=strategy)
    @settings(max_examples=3, deadline=None)
    def check(obj: object) -> None:
        assert isinstance(obj, cls)

    check()  # raises Unsatisfiable if `cls` has no registered strategy


class TestOnlyDrawableClassesAreOffered:
    """The candidate pool never offers a class the redispatch cannot draw.

    ``NoManifold``/``MinkowskiManifold`` (and ``NoAtlas``/``MinkowskiAtlas``)
    have no strategy registered, so every example that selected one was thrown
    away. At ``ndim=5``, where two of the three manifold candidates were such
    types, that was enough filtering to trip the ``filter_too_much`` health
    check on unlucky seeds.
    """

    @pytest.mark.parametrize("ndim", [None, 0, 1, 2, 3, 4, 5, 6])
    def test_every_manifold_candidate_is_drawable(self, ndim: int | None) -> None:
        """Each class the pool can select yields an instance, not a discard."""
        for cls in get_all_subclasses(cxm.AbstractManifold, exclude_abstract=True):
            if _manifold_class_supports_ndim(cls, ndim):
                _assert_drawable(cxst.manifolds(cls, ndim=ndim), cls)

    @pytest.mark.parametrize("ndim", [None, 0, 1, 2, 3])
    def test_every_atlas_candidate_is_drawable(self, ndim: int | None) -> None:
        """Same invariant one level down, for the atlas pool."""
        for cls in get_all_subclasses(cxm.AbstractAtlas, exclude_abstract=True):
            if _atlas_class_supports_ndim(cls, ndim):
                _assert_drawable(cxst.atlases(cls, ndim=ndim), cls)
