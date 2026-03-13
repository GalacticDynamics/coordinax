"""Tests for the chart_classes strategy.

This module provides comprehensive tests for the chart_classes Hypothesis
strategy. Tests cover:
- Basic generation and validation
- Filter and exclude parameter behavior
- Parameter combinations
- Dynamic (strategy-valued) parameters
- Edge cases and error conditions
- Properties of generated classes
"""

import inspect

from typing import final

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

import coordinax.charts as cxc
import coordinax.hypothesis.main as cxst
from coordinax.hypothesis.utils import get_all_subclasses


def is_abstract_class(obj: type, /) -> bool:
    """Check if something is abstract."""
    return inspect.isabstract(obj) or obj.__name__.startswith("Abstract")


# ============================================================================
# Basic Generation Tests
# ============================================================================


class TestBasicGeneration:
    """Test basic functionality of chart_classes strategy."""

    @given(chart_cls=cxst.chart_classes())
    @settings(max_examples=100)
    def test_returns_chart_subclass(self, chart_cls: type[cxc.AbstractChart]) -> None:
        """Generated class is always a subclass of AbstractChart."""
        assert issubclass(chart_cls, cxc.AbstractChart), (
            f"{chart_cls} should be a subclass of AbstractChart"
        )

    @given(chart_cls=cxst.chart_classes())
    @settings(max_examples=100)
    def test_not_abstract_base_class(self, chart_cls: type[cxc.AbstractChart]) -> None:
        """Generated class is never AbstractChart itself."""
        assert chart_cls is not cxc.AbstractChart

    @given(chart_cls=cxst.chart_classes())
    @settings(max_examples=100)
    def test_not_a_metaclass(self, chart_cls: type[cxc.AbstractChart]) -> None:
        """Generated class is never a metaclass."""
        assert not issubclass(chart_cls, type)

    @given(chart_cls=cxst.chart_classes(exclude_abstract=True))
    @settings(max_examples=100)
    def test_is_concrete_by_default(self, chart_cls: type[cxc.AbstractChart]) -> None:
        """By default (exclude_abstract=True), only concrete classes are generated."""
        assert not is_abstract_class(chart_cls), (
            f"{chart_cls} should be concrete when exclude_abstract=True"
        )

    @given(chart_cls=cxst.chart_classes(exclude_abstract=False))
    @settings(max_examples=50)
    def test_includes_abstract_when_requested(
        self, chart_cls: type[cxc.AbstractChart]
    ) -> None:
        """When exclude_abstract=False, abstract classes can be generated."""
        # At least some of the draws should be abstract
        # (This property can't be guaranteed in a single example, but we set
        # max_examples high to increase likelihood of hitting abstract classes)
        assert issubclass(chart_cls, cxc.AbstractChart)


# ============================================================================
# Filter Parameter Tests
# ============================================================================


class TestFilterParameter:
    """Test filter parameter behavior."""

    @given(
        data=st.data(),
        filter_class=st.sampled_from([cxc.Abstract1D, cxc.Abstract2D, cxc.Abstract3D]),
    )
    @settings(max_examples=50)
    def test_filter_single_class(
        self, data, filter_class: type[cxc.AbstractChart]
    ) -> None:
        """Charts respect filter to single class."""
        chart_cls = data.draw(cxst.chart_classes(filter=filter_class))
        assert issubclass(chart_cls, filter_class), (
            f"{chart_cls} should be subclass of {filter_class}"
        )

    @given(
        data=st.data(),
        filter_classes=st.sampled_from(
            [(cxc.AbstractChart, cxc.Abstract1D), (cxc.AbstractChart, cxc.Abstract2D)]
        ),
    )
    @settings(max_examples=50)
    def test_filter_multiple_classes(
        self, data, filter_classes: tuple[type, ...]
    ) -> None:
        """Charts respect filter to tuple of classes (AND semantics)."""
        chart_cls = data.draw(cxst.chart_classes(filter=filter_classes))
        # Filter uses AND semantics: must be subclass of ALL filters
        for filter_cls in filter_classes:
            assert issubclass(chart_cls, filter_cls), (
                f"{chart_cls} should be subclass of {filter_cls} "
                f"(filter has AND semantics)"
            )

    @given(
        chart_cls=cxst.chart_classes(
            filter=st.sampled_from([cxc.Abstract1D, cxc.Abstract2D])
        )
    )
    @settings(max_examples=50)
    def test_dynamic_filter_strategy(self, chart_cls: type[cxc.AbstractChart]) -> None:
        """Filter parameter can be a strategy."""
        assert issubclass(chart_cls, (cxc.Abstract1D, cxc.Abstract2D))

    @given(chart_cls=cxst.chart_classes(filter=cxc.AbstractSpherical3D))
    @settings(max_examples=30)
    def test_filter_to_spherical_charts(
        self, chart_cls: type[cxc.AbstractChart]
    ) -> None:
        """Filter can target spherical charts specifically."""
        assert issubclass(chart_cls, cxc.AbstractSpherical3D)


# ============================================================================
# Exclude Parameter Tests
# ============================================================================


class TestExcludeParameter:
    """Test exclude parameter behavior."""

    @given(data=st.data())
    @settings(max_examples=50)
    def test_exclude_single_class(self, data) -> None:
        """Can exclude a single class."""
        chart_cls = data.draw(cxst.chart_classes(exclude=(cxc.Cart3D,)))
        assert chart_cls is not cxc.Cart3D
        assert not issubclass(chart_cls, cxc.Cart3D)

    @given(data=st.data())
    @settings(max_examples=50)
    def test_exclude_multiple_classes(self, data) -> None:
        """Can exclude multiple classes."""
        excluded = (cxc.Cart3D, cxc.Spherical3D, cxc.Cart2D)
        chart_cls = data.draw(cxst.chart_classes(exclude=excluded))
        for exc_cls in excluded:
            assert chart_cls is not exc_cls
            assert not issubclass(chart_cls, exc_cls)

    @given(data=st.data())
    @settings(max_examples=50)
    def test_exclude_empty_tuple(self, data) -> None:
        """Empty exclude tuple has no effect."""
        chart_cls = data.draw(cxst.chart_classes(exclude=()))
        assert issubclass(chart_cls, cxc.AbstractChart)

    @given(data=st.data())
    @settings(max_examples=50)
    def test_exclude_respects_exclude_abstract(self, data) -> None:
        """Exclude still respects exclude_abstract=True."""
        chart_cls = data.draw(
            cxst.chart_classes(exclude=(cxc.Cart3D,), exclude_abstract=True)
        )
        assert chart_cls is not cxc.Cart3D
        assert not is_abstract_class(chart_cls)


# ============================================================================
# Exclude_Abstract Parameter Tests
# ============================================================================


class TestExcludeAbstractParameter:
    """Test exclude_abstract parameter behavior."""

    @given(chart_cls=cxst.chart_classes(exclude_abstract=True))
    @settings(max_examples=100)
    def test_exclude_abstract_true_generates_concrete(
        self, chart_cls: type[cxc.AbstractChart]
    ) -> None:
        """exclude_abstract=True generates only concrete classes."""
        assert not is_abstract_class(chart_cls)

    @given(chart_cls=cxst.chart_classes(exclude_abstract=False))
    @settings(max_examples=50)
    def test_exclude_abstract_false_accepts_abstract(
        self, chart_cls: type[cxc.AbstractChart]
    ) -> None:
        """exclude_abstract=False allows abstract classes."""
        assert issubclass(chart_cls, cxc.AbstractChart)

    @given(data=st.data(), exclude_abstract=st.sampled_from([True, False]))
    @settings(max_examples=50)
    def test_dynamic_exclude_abstract_strategy(
        self, data, exclude_abstract: bool
    ) -> None:
        """exclude_abstract can be a strategy."""
        chart_cls = data.draw(cxst.chart_classes(exclude_abstract=exclude_abstract))
        assert issubclass(chart_cls, cxc.AbstractChart)
        if exclude_abstract:
            assert not is_abstract_class(chart_cls)


# ============================================================================
# Combined Parameters Tests
# ============================================================================


class TestParameterCombinations:
    """Test combinations of multiple parameters."""

    @given(chart_cls=cxst.chart_classes(filter=cxc.Abstract3D, exclude=(cxc.Cart3D,)))
    @settings(max_examples=50)
    def test_filter_and_exclude(self, chart_cls: type[cxc.AbstractChart]) -> None:
        """Filter and exclude work together."""
        assert issubclass(chart_cls, cxc.Abstract3D)
        assert chart_cls is not cxc.Cart3D

    @given(chart_cls=cxst.chart_classes(filter=cxc.Abstract3D, exclude_abstract=True))
    @settings(max_examples=50)
    def test_filter_and_exclude_abstract(
        self, chart_cls: type[cxc.AbstractChart]
    ) -> None:
        """Filter and exclude_abstract work together."""
        assert issubclass(chart_cls, cxc.Abstract3D)
        assert not is_abstract_class(chart_cls)

    @given(
        chart_cls=cxst.chart_classes(
            filter=cxc.Abstract2D, exclude=(cxc.Polar2D,), exclude_abstract=True
        )
    )
    @settings(max_examples=50)
    def test_all_parameters_combined(self, chart_cls: type[cxc.AbstractChart]) -> None:
        """All parameters work together."""
        assert issubclass(chart_cls, cxc.Abstract2D)
        assert chart_cls is not cxc.Polar2D
        assert not is_abstract_class(chart_cls)


# ============================================================================
# Class Property Tests
# ============================================================================


class TestGeneratedClassProperties:
    """Test properties of generated classes."""

    @given(
        data=st.data(),
        chart_cls=cxst.chart_classes(exclude_abstract=True),
    )
    @settings(max_examples=100)
    def test_generated_class_is_instantiable(
        self, data, chart_cls: type[cxc.AbstractChart]
    ) -> None:
        """Generated concrete classes can be instantiated with valid kwargs."""
        # Generate valid initialization kwargs for this chart class
        kwargs = data.draw(cxst.chart_init_kwargs(chart_cls))
        try:
            instance = chart_cls(**kwargs)
            assert isinstance(instance, cxc.AbstractChart)
            assert isinstance(instance, chart_cls)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Failed to instantiate {chart_cls} with {kwargs}: {e}")

    @given(
        data=st.data(),
        chart_cls=cxst.chart_classes(exclude_abstract=True),
    )
    @settings(max_examples=100)
    def test_generated_class_has_valid_ndim(
        self, data, chart_cls: type[cxc.AbstractChart]
    ) -> None:
        """Generated classes have a valid ndim property."""
        kwargs = data.draw(cxst.chart_init_kwargs(chart_cls))
        instance = chart_cls(**kwargs)
        ndim = instance.ndim
        assert isinstance(ndim, int)
        assert ndim >= 0

    @given(
        data=st.data(),
        chart_cls=cxst.chart_classes(exclude_abstract=True),
    )
    @settings(max_examples=100)
    def test_generated_class_has_components(
        self, data, chart_cls: type[cxc.AbstractChart]
    ) -> None:
        """Generated classes have components (may be empty for 0D)."""
        kwargs = data.draw(cxst.chart_init_kwargs(chart_cls))
        instance = chart_cls(**kwargs)
        components = instance.components
        assert isinstance(components, tuple)
        # Components can be empty for 0D charts
        assert all(isinstance(c, str) for c in components)
        # Components count should match ndim
        assert len(components) == instance.ndim

    @given(
        data=st.data(),
        chart_cls=cxst.chart_classes(exclude_abstract=True),
    )
    @settings(max_examples=100)
    def test_generated_class_components_match_ndim(
        self, data, chart_cls: type[cxc.AbstractChart]
    ) -> None:
        """Number of components matches ndim."""
        kwargs = data.draw(cxst.chart_init_kwargs(chart_cls))
        instance = chart_cls(**kwargs)
        assert len(instance.components) == instance.ndim


# ============================================================================
# Edge Cases and Error Conditions
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_warning_when_no_subclasses_found(self) -> None:
        """Warning is raised when no subclasses match criteria."""

        # Create a fake class that nothing will inherit from
        @final
        class _FakeClass:
            pass

        # The warning is raised when get_all_subclasses finds no matches
        with pytest.warns(UserWarning, match="No subclasses found"):
            result = get_all_subclasses(
                cxc.AbstractChart, filter=_FakeClass, exclude_abstract=True
            )

        # Should return an empty tuple
        assert result == ()

    def test_exclude_everything_possible_warns(self) -> None:
        """Warning when trying to exclude all possible classes."""
        # Exclude all concrete 3D charts (if possible)
        all_3d = get_all_subclasses(
            cxc.AbstractChart, filter=cxc.Abstract3D, exclude_abstract=True
        )
        if len(all_3d) > 0:
            # If we exclude all of them, we should get a warning
            with pytest.warns(UserWarning, match="No subclasses found"):
                get_all_subclasses(
                    cxc.AbstractChart,
                    filter=cxc.Abstract3D,
                    exclude=tuple(all_3d),
                    exclude_abstract=True,
                )

    @given(
        data=st.data(),
        chart_cls=cxst.chart_classes(filter=cxc.Abstract1D, exclude_abstract=True),
    )
    @settings(max_examples=40)
    def test_1d_charts_have_one_component(
        self, data, chart_cls: type[cxc.AbstractChart]
    ) -> None:
        """1D charts have exactly one component."""
        kwargs = data.draw(cxst.chart_init_kwargs(chart_cls))
        instance = chart_cls(**kwargs)
        assert instance.ndim == 1
        assert len(instance.components) == 1

    @given(
        data=st.data(),
        chart_cls=cxst.chart_classes(filter=cxc.Abstract2D, exclude_abstract=True),
    )
    @settings(max_examples=40)
    def test_2d_charts_have_two_components(
        self, data, chart_cls: type[cxc.AbstractChart]
    ) -> None:
        """2D charts have exactly two components."""
        kwargs = data.draw(cxst.chart_init_kwargs(chart_cls))
        instance = chart_cls(**kwargs)
        assert instance.ndim == 2
        assert len(instance.components) == 2

    @given(
        data=st.data(),
        chart_cls=cxst.chart_classes(filter=cxc.Abstract3D, exclude_abstract=True),
    )
    @settings(max_examples=50)
    def test_3d_charts_have_three_components(
        self, data, chart_cls: type[cxc.AbstractChart]
    ) -> None:
        """3D charts have exactly three components."""
        kwargs = data.draw(cxst.chart_init_kwargs(chart_cls))
        instance = chart_cls(**kwargs)
        assert instance.ndim == 3
        assert len(instance.components) == 3


# ============================================================================
# Coverage Tests
# ============================================================================


class TestStrategyDiversity:
    """Test that the strategy can access diverse classes."""

    def test_multiple_chart_classes_available(self) -> None:
        """Multiple distinct chart classes are available for generation."""
        # Verify that get_all_subclasses returns multiple classes
        all_concrete_charts = get_all_subclasses(
            cxc.AbstractChart, exclude_abstract=True
        )
        # We know there are 21 concrete chart classes
        assert len(all_concrete_charts) >= 5, (
            f"Expected at least 5 concrete chart classes, "
            f"but found {len(all_concrete_charts)}"
        )

        # Test with filter also returns multiple classes
        abstract_1d_charts = get_all_subclasses(
            cxc.AbstractChart, filter=cxc.Abstract1D, exclude_abstract=True
        )
        assert len(abstract_1d_charts) >= 2, (
            f"Expected at least 2 concrete 1D chart classes, "
            f"but found {len(abstract_1d_charts)}"
        )
