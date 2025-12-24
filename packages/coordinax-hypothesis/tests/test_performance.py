"""Performance tests to verify strategies are reasonably fast."""

import time

import pytest

import coordinax as cx
import coordinax_hypothesis as cxst


def test_representations_strategy_performance():
    """Test that charts() strategy completes in reasonable time.

    This test verifies that the caching optimizations are working.
    With caching, building strategies should be fast (< 100ms for 100 iterations).
    """
    iterations = 100
    start = time.perf_counter()

    for _ in range(iterations):
        # Build the strategy (this is what we optimized with caching)
        _ = cxst.charts()
        # Don't draw from it - that's hypothesis internals

    elapsed = time.perf_counter() - start

    # With caching, this should be very fast (< 1ms per iteration)
    # Allow up to 100ms total for 100 iterations to account for CI variability
    assert elapsed < 0.1, (
        f"Strategy building took {elapsed:.3f}s for {iterations} iterations"
    )


def test_chart_classes_performance():
    """Test that chart_classes() strategy builds quickly."""
    iterations = 100
    start = time.perf_counter()

    for _ in range(iterations):
        _ = cxst.chart_classes()

    elapsed = time.perf_counter() - start

    # Should be even faster than charts()
    assert elapsed < 0.1, (
        f"Strategy building took {elapsed:.3f}s for {iterations} iterations"
    )


@pytest.mark.filterwarnings("ignore::hypothesis.errors.NonInteractiveExampleWarning")
def test_drawing_representations_is_fast():
    """Test that we can draw multiple examples without excessive delay.

    This tests the end-to-end performance of the strategy.
    """
    strategy = cxst.charts()

    start = time.perf_counter()
    examples = [strategy.example() for _ in range(10)]
    elapsed = time.perf_counter() - start

    # Drawing 10 examples should complete in < 1 second
    assert elapsed < 1.0, f"Drawing 10 examples took {elapsed:.3f}s"

    # Verify we got valid representations
    assert all(isinstance(rep, cx.charts.AbstractChart) for rep in examples)
