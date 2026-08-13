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

import unxts.hypothesis as ust

import coordinax.charts as cxc
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
