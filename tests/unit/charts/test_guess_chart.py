"""Tests for guess_chart function.

`guess_chart` has fundamental limitations:

1. It only works with charts that can be instantiated without arguments.
   Charts like ProlateSpheroidal3D that require constructor arguments are
   not discoverable.

2. When multiple chart types share the same component names (e.g.,
   Spherical3D and MathSpherical3D both use ('r', 'theta', 'phi')),
   `guess_chart` has to pick one: component names alone don't uniquely
   identify chart types. `CANONICAL_CHART_CLASSES` names which, and
   `TestAmbiguousComponentNames` pins it -- the alternatives disagree about
   what ``theta`` means, so the choice is a numerical answer, not a detail.

"""

from typing import ClassVar, TypeAlias

import numpy as np
import pytest
from hypothesis import given, strategies as st

import unxt as u
import unxts.hypothesis as ust

import coordinax.charts as cxc
import coordinax.manifolds as cxm
import coordinaxs.hypothesis.main as cxst
from .conftest import SHAPE_CART_MAP, xps
from coordinax._src.base import NON_ABC_CHART_CLASSES, AbstractFixedComponentsChart
from coordinax._src.charts.register_guess import (
    CANONICAL_CHART_CLASSES,
    guess_chart_cls,
    register_canonical_chart,
)

Shape: TypeAlias = tuple[int, ...]


def is_guessable(chart: cxc.AbstractChart) -> bool:
    """Check if a chart can be recovered by guess_chart.

    A chart is guessable if its type can be instantiated without arguments.
    """
    try:
        type(chart)()
    except TypeError:
        return False
    return True


guessable_charts = cxst.charts(filter=cxc.AbstractFixedComponentsChart).filter(
    is_guessable
)


@given(guessable_charts)
def test_guess_chart_returns_same_components(
    chart: cxc.AbstractFixedComponentsChart,
) -> None:
    """guess_chart(frozenset(chart.components)) returns chart with same components.

    Note: We test component equality, not type equality, because multiple
    chart types can share the same component names (e.g., Spherical3D and
    MathSpherical3D both use 'r', 'theta', 'phi').
    """
    # Guess the chart from the components
    guessed = cxc.guess_chart(frozenset(chart.components))

    # The guessed chart should have the same components
    # Note: We only test component equality because multiple chart types can
    # share the same component names (e.g., Spherical3D and MathSpherical3D)
    assert guessed.components == chart.components


@given(guessable_charts)
def test_guess_chart_from_dict_returns_same_components(
    chart: cxc.AbstractFixedComponentsChart,
) -> None:
    """guess_chart with dict input returns chart with same components."""
    # Create a component dictionary with dummy values
    d = dict.fromkeys(chart.components, 1)

    # Guess the chart from the dict
    guessed = cxc.guess_chart(d)

    # The guessed chart should have the same components
    assert guessed.components == chart.components


class TestGuessChartRepeatability:
    """Repeated inference agrees.

    Note it is *not* cached: each call builds a chart, so the results are equal
    but not identical.
    """

    def test_frozenset_dispatch(self) -> None:
        """The frozenset dispatch should return equal objects."""
        keys = frozenset(("x", "y", "z"))
        result1 = cxc.guess_chart(keys)
        result2 = cxc.guess_chart(keys)
        assert result1 == result2

    def test_dict_dispatch_returns_same_type(self) -> None:
        """The dict dispatch should return same chart type for same keys."""
        d1 = {"x": 1, "y": 2, "z": 3}
        d2 = {"x": 5, "y": 6, "z": 7}
        result1 = cxc.guess_chart(d1)
        result2 = cxc.guess_chart(d2)
        assert type(result1) is type(result2)


class TestGuessChartFailureModes:
    """No chart matches, and the messages that come back."""

    def test_unknown_component_names_raise(self) -> None:
        """The spec's failure semantics: an unmatched key set is a `ValueError`."""
        with pytest.raises(ValueError, match="Cannot infer representation"):
            cxc.guess_chart(frozenset(("nope", "nada")))

    def test_the_message_lists_the_keys_in_a_stable_order(self) -> None:
        """`frozenset` iterates in `id`-hash order; the message must not."""
        with pytest.raises(ValueError, match=r"\['nada', 'nope'\]"):
            guess_chart_cls(frozenset(("nope", "nada")))


class TestAmbiguousComponentNames:
    """Charts sharing a component-name set resolve to one declared chart.

    `guess_chart` scans `NON_ABC_CHART_CLASSES`, a `weakref.WeakSet`, which
    iterates in `id`-hash order and so differs between processes. What makes
    the answer a function of the component names is `CANONICAL_CHART_CLASSES`,
    which names the chart to infer; the alternatives disagree about which angle
    is polar, so picking the other one silently moves the point.
    """

    #: Component-name sets shared by several charts, and the one to infer.
    #: Both pairs are physics convention vs maths convention.
    AMBIGUOUS: ClassVar = [
        pytest.param(("r", "theta", "phi"), "Spherical3D", id="sph3d"),
        pytest.param(("theta", "phi"), "SphericalTwoSphere", id="sph2"),
    ]

    def test_every_ambiguous_component_set_is_declared(self) -> None:
        """The invariant `guess_chart_cls` relies on, over every chart."""
        by_components: dict[frozenset[str], list[str]] = {}
        for cls in NON_ABC_CHART_CLASSES:
            if issubclass(cls, AbstractFixedComponentsChart):
                by_components.setdefault(frozenset(cls._components), []).append(
                    cls.__name__
                )

        ambiguous = {k for k, v in by_components.items() if len(v) > 1}
        undeclared = {
            tuple(sorted(k)): sorted(by_components[k])
            for k in ambiguous - set(CANONICAL_CHART_CLASSES)
        }
        assert not undeclared, (
            f"Component names no longer identify a chart: {undeclared}. Add "
            "the intended one with `register_canonical_chart`."
        )

    @pytest.mark.parametrize(("components", "expected"), AMBIGUOUS)
    def test_ambiguity_resolves_to_the_declared_chart(
        self, components: tuple[str, ...], expected: str
    ) -> None:
        """The physics-convention chart is inferred, in every process."""
        assert type(cxc.guess_chart(frozenset(components))).__name__ == expected

    @pytest.mark.parametrize(("components", "expected"), AMBIGUOUS)
    def test_the_declaration_names_that_chart(
        self, components: tuple[str, ...], expected: str
    ) -> None:
        """The choice is declared, not a by-product of scan order."""
        assert CANONICAL_CHART_CLASSES[frozenset(components)].__name__ == expected

    def test_redeclaring_the_same_chart_is_a_no_op(self) -> None:
        """A module may be imported twice without tripping the guard."""
        before = dict(CANONICAL_CHART_CLASSES)
        register_canonical_chart(cxc.Spherical3D)
        assert before == CANONICAL_CHART_CLASSES

    def test_a_second_chart_cannot_claim_the_same_names(self) -> None:
        """Overwriting would hand the choice back to import order."""
        with pytest.raises(ValueError, match="already declared canonical"):
            register_canonical_chart(cxc.MathSpherical3D)
        assert CANONICAL_CHART_CLASSES[frozenset(("r", "theta", "phi"))] is (
            cxc.Spherical3D
        )

    def test_unresolvable_ambiguity_names_the_candidates(self) -> None:
        """An undeclared collision raises rather than picking one."""
        keys = frozenset(("r", "theta", "phi"))
        canonical = CANONICAL_CHART_CLASSES.pop(keys)
        try:
            with pytest.raises(ValueError, match="none of them is canonical"):
                guess_chart_cls(keys)
        finally:
            CANONICAL_CHART_CLASSES[keys] = canonical


class TestGuessChartFromArrayLike:
    """Test guess_chart with array/quantity inputs."""

    def draw_shape(self, data, ndim: int) -> Shape:
        """Draw shape."""
        return data.draw(
            xps.array_shapes(max_dims=3, max_side=3).map(lambda s: (*s, ndim))
        )

    @given(data=st.data(), ndim=st.sampled_from([1, 2, 3]))
    def test_array_trailing_dim_guesses_cartesian(
        self, data: st.DataObject, ndim: int
    ) -> None:
        """Arrays with shape (*batch, ndim) should guess to corresponding Cart[N]D."""
        expected = SHAPE_CART_MAP[ndim]
        arr = data.draw(
            xps.arrays(dtype=xps.real_dtypes(), shape=self.draw_shape(data, ndim))
        )

        guessed = cxc.guess_chart(arr)
        assert guessed == expected

    @given(data=st.data(), ndim=st.sampled_from([1, 2, 3]))
    def test_quantity_trailing_dim_guesses_cartesian(
        self, data: st.DataObject, ndim: int
    ) -> None:
        """Quantities with shape (*batch, ndim) should guess to Cart[N]D."""
        expected = SHAPE_CART_MAP[ndim]
        q = data.draw(ust.quantities("m", shape=self.draw_shape(data, ndim)))
        guessed = cxc.guess_chart(q)
        assert guessed == expected

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_numpy_array_trailing_dim_guesses_cartesian(self, ndim: int) -> None:
        """A NumPy array (not only a JAX array) guesses to Cart[N]D."""
        guessed = cxc.guess_chart(np.ones((2, ndim)))
        assert guessed == SHAPE_CART_MAP[ndim]

    def test_numpy_array_high_dim_guesses_cartnd(self) -> None:
        """A NumPy array with trailing dim > 3 guesses to the N-D Cartesian chart."""
        guessed = cxc.guess_chart(np.ones((2, 5)))
        assert type(guessed) is cxc.CartND


class TestGuessManifoldOnChartClasses:
    """`guess_chart` dispatches `guess_manifold` on the chart *class*.

    The `type[AbstractChart]` fallback returns `no_manifold` silently, so a
    chart class that never declared a rule produced instances carrying
    `NoManifold()` and nothing raised. That is how `guess_chart({"theta": ...,
    "phi": ...})` came to return `SphericalTwoSphere(M=NoManifold())` while the
    same class default-constructed to `HyperSphericalManifold(ndim=2)`.
    """

    #: Classes that genuinely cannot fix a manifold from the class alone.
    #: `CartND` carries its dimension per instance; `PoincarePolar6D` has no
    #: manifold even as an instance.
    NO_CLASS_LEVEL_MANIFOLD: ClassVar[set[str]] = {"CartND", "PoincarePolar6D"}

    @staticmethod
    def _default_constructible() -> list[type]:
        out = []
        for cls in NON_ABC_CHART_CLASSES:
            if not issubclass(cls, AbstractFixedComponentsChart):
                continue
            try:
                cls()
            except Exception:  # noqa: BLE001, S112  # needs constructor arguments
                continue
            out.append(cls)
        return sorted(out, key=lambda c: c.__name__)

    def test_class_level_matches_instance_level(self) -> None:
        """`guess_manifold(cls)` must agree with `cls().M`.

        Asserting agreement rather than a hard-coded table, so a new chart is
        covered the day it is added instead of the day someone remembers to
        extend a list.
        """
        mismatched = {}
        for cls in self._default_constructible():
            if cls.__name__ in self.NO_CLASS_LEVEL_MANIFOLD:
                continue
            from_class = cxm.guess_manifold(cls)
            from_instance = cls().M
            if from_class != from_instance:
                mismatched[cls.__name__] = (from_class, from_instance)
        assert not mismatched, (
            f"class-level guess disagrees with instance: {mismatched}"
        )

    def test_no_chart_class_silently_yields_no_manifold(self) -> None:
        """The fallback must be reached only by the classes named above."""
        fell_back = {
            cls.__name__
            for cls in self._default_constructible()
            if isinstance(cxm.guess_manifold(cls), cxm.NoManifold)
        }
        assert fell_back == self.NO_CLASS_LEVEL_MANIFOLD

    def test_guessed_chart_carries_its_manifold(self) -> None:
        """The user-visible symptom: an inferred chart must not carry the sentinel."""
        assert cxc.guess_chart({"theta": 1.0, "phi": 0.5}).M == cxm.S2
        assert cxm.Rn(1) == cxc.guess_chart({"t": u.Q(1.0, "s")}).M
        assert cxm.guess_manifold({"theta": 1.0, "phi": 0.5}) == cxm.S2
