"""Tests for guess_chart function.

Notes
-----
`guess_chart` has fundamental limitations:

1. It only works with charts that can be instantiated without arguments.
   Charts like ProlateSpheroidal3D that require constructor arguments are
   not discoverable.

2. When multiple chart types share the same component names (e.g.,
   Spherical3D and MathSpherical3D both use ('r', 'theta', 'phi')),
   `guess_chart` returns whichever it finds first in iteration order.
   This is by design - component names alone don't uniquely identify
   chart types.

"""

from hypothesis import given, settings, strategies as st

import unxt_hypothesis as ust

import coordinax.charts as cxc
import coordinax_hypothesis as cxst
from .conftest import SHAPE_CART_MAP, xps
from coordinax._src.custom_types import Shape


def is_guessable(chart: cxc.AbstractChart) -> bool:  # type: ignore[type-arg]
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
@settings(max_examples=100)
def test_guess_chart_returns_same_components(
    chart: cxc.AbstractFixedComponentsChart,  # type: ignore[type-arg]
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
@settings(max_examples=100)
def test_guess_chart_from_dict_returns_same_components(
    chart: cxc.AbstractFixedComponentsChart,  # type: ignore[type-arg]
) -> None:
    """guess_chart with dict input returns chart with same components."""
    # Create a component dictionary with dummy values
    d = dict.fromkeys(chart.components, 1.0)

    # Guess the chart from the dict
    guessed = cxc.guess_chart(d)

    # The guessed chart should have the same components
    assert guessed.components == chart.components


class TestGuessChartCaching:
    """Test that guess_chart caching works correctly."""

    def test_frozenset_dispatch_is_cached(self) -> None:
        """The frozenset dispatch should return the same object on repeated calls."""
        keys = frozenset(("x", "y", "z"))
        result1 = cxc.guess_chart(keys)
        result2 = cxc.guess_chart(keys)
        # Same object due to @ft.cache
        assert result1 is result2

    def test_dict_dispatch_returns_same_type(self) -> None:
        """The dict dispatch should return same chart type for same keys."""
        d1 = {"x": 1.0, "y": 2.0, "z": 3.0}
        d2 = {"x": 5.0, "y": 6.0, "z": 7.0}
        result1 = cxc.guess_chart(d1)
        result2 = cxc.guess_chart(d2)
        assert type(result1) is type(result2)


class TestGuessChartFromArrayLike:
    """Test guess_chart with array/quantity inputs."""

    def draw_shape(self, data, ndim: int) -> Shape:
        """Draw shape."""
        return data.draw(
            xps.array_shapes(max_dims=3, max_side=3).map(lambda s: (*s, ndim))
        )

    @given(data=st.data(), ndim=st.sampled_from([1, 2, 3]))
    @settings(max_examples=100)
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
    @settings(max_examples=100)
    def test_quantity_trailing_dim_guesses_cartesian(
        self, data: st.DataObject, ndim: int
    ) -> None:
        """Quantities with shape (*batch, ndim) should guess to Cart[N]D."""
        expected = SHAPE_CART_MAP[ndim]
        q = data.draw(ust.quantities("m", shape=self.draw_shape(data, ndim)))
        guessed = cxc.guess_chart(q)
        assert guessed == expected
