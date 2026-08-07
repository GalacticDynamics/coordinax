"""Tests for manifold strategies."""

from collections.abc import Callable

import hypothesis.strategies as st
import pytest
from hypothesis import find, given, settings
from hypothesis.errors import Unsatisfiable

import coordinax.charts as cxc
import coordinax.manifolds as cxm

import coordinaxs.hypothesis.main as cxst
from coordinaxs.hypothesis.manifolds._src.atlas import _atlas_class_supports_ndim
from coordinaxs.hypothesis.manifolds._src.manifold import _manifold_class_supports_ndim


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

    @pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5, 6, 7])
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


class TestOnlyDrawableCandidatesAreSelected:
    """Classes with no registered strategy are never offered as candidates."""

    @pytest.mark.parametrize(
        ("cls", "strategy"),
        [
            (cxm.NoAtlas, cxst.atlases),
            (cxm.MinkowskiAtlas, cxst.atlases),
            (cxm.NoManifold, cxst.manifolds),
            (cxm.MinkowskiManifold, cxst.manifolds),
        ],
    )
    @given(data=st.data())
    def test_requesting_one_directly_says_so(
        self,
        cls: type,
        strategy: Callable[..., st.SearchStrategy[object]],
        data: st.DataObject,
    ) -> None:
        """Asking for one by name raises, naming the class."""
        with pytest.raises(NotImplementedError, match=cls.__name__):
            data.draw(strategy(cls))

    @given(atlas=cxst.atlases())
    def test_drawn_atlases_all_have_strategies(self, atlas: object) -> None:
        """No draw yields a type the module has no strategy for."""
        assert not isinstance(atlas, (cxm.NoAtlas, cxm.MinkowskiAtlas))

    @given(M=cxst.manifolds())
    def test_drawn_manifolds_all_have_strategies(self, M: object) -> None:
        """Same for manifolds."""
        assert not isinstance(M, (cxm.NoManifold, cxm.MinkowskiManifold))

    @pytest.mark.parametrize(
        ("ndim", "supported"),
        [(1, True), (2, True), (3, True), (4, True), (5, False), (7, False)],
    )
    def test_custom_atlas_support_tracks_available_charts(
        self, ndim: int, supported: bool
    ) -> None:
        """``CustomAtlas`` supports only dimensionalities with zero-arg charts.

        Spelled out, not recomputed from `matching_chart_classes_for_ndim` --
        that is what the predicate calls, so it would assert itself.
        """
        assert _atlas_class_supports_ndim(cxm.CustomAtlas, ndim) is supported

    @pytest.mark.parametrize(
        ("supports_ndim", "cls"),
        [
            (_atlas_class_supports_ndim, cxm.MinkowskiAtlas),
            (_manifold_class_supports_ndim, cxm.MinkowskiManifold),
        ],
    )
    def test_types_absent_from_the_table_default_to_supported(
        self, supports_ndim: Callable[[type, int], bool], cls: type
    ) -> None:
        """A type with no `_NDIM_SUPPORT` entry is treated as unrestricted.

        Which is why `_NO_STRATEGY` has to be a separate gate: this predicate
        answers "at which ndim", not "can it be drawn at all", and would wave
        these two through at every ndim.
        """
        assert supports_ndim(cls, 3) is True


class TestFactorCountIsDrawnFeasible:
    """``n_factors`` is drawn from the feasible range, not drawn then rejected.

    Both product strategies used to draw 1-5 factors and `assume` the count
    against the target. That discarded ``1 - feasible/5`` of all draws -- four
    in five at ``ndim=1`` -- and biased the survivors toward few-factor
    products, since a small target only ever admits the small counts.
    """

    @pytest.mark.parametrize(("ndim", "counts"), [(1, [1]), (2, [1, 2]), (5, [1, 5])])
    def test_manifold_factor_counts_are_reachable(
        self, ndim: int, counts: list[int]
    ) -> None:
        """Every factor count a product of this ``ndim`` admits is drawable."""
        strategy = cxst.manifolds(cxm.CartesianProductManifold, ndim=ndim)
        for n in counts:
            find(strategy, lambda M, n=n: len(M.factors) == n)

    @pytest.mark.parametrize(("ndim", "counts"), [(1, [1]), (6, [2, 5]), (15, [5])])
    def test_atlas_factor_counts_are_reachable(
        self, ndim: int, counts: list[int]
    ) -> None:
        """Same for atlases, whose factors are capped at 3 dimensions each.

        ``ndim=6`` cannot be a 1-factor product (no 6-D factor atlas) and
        ``ndim=15`` only fits as 5 factors of 3 -- the bounds the ``lo``/``hi``
        arithmetic has to get right.
        """
        strategy = cxst.atlases(cxm.CartesianProductAtlas, ndim=ndim)
        for n in counts:
            find(strategy, lambda a, n=n: len(a.factors) == n)

    @pytest.mark.parametrize(
        ("strategy_for", "cls", "ndim"),
        [
            (cxst.manifolds, cxm.CartesianProductManifold, 0),
            (cxst.manifolds, cxm.CartesianProductManifold, -1),
            (cxst.atlases, cxm.CartesianProductAtlas, 0),
            (cxst.atlases, cxm.CartesianProductAtlas, 16),
        ],
    )
    def test_unreachable_ndim_is_unsatisfiable(
        self, strategy_for: Callable[..., st.SearchStrategy], cls: type, ndim: int
    ) -> None:
        """An ``ndim`` no factor count reaches is discarded, never clamped.

        The atlas ceiling is 15: five factors of at most 3 dimensions each.
        """

        @given(obj=strategy_for(cls, ndim=ndim))
        @settings(max_examples=5, deadline=None)
        def check(obj: object) -> None:
            pytest.fail(f"ndim={ndim} should be unsatisfiable, got {obj!r}")

        with pytest.raises(Unsatisfiable):
            check()
