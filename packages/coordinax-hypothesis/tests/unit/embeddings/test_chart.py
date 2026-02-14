"""Tests for the chart_init_kwargs strategy."""

from hypothesis import given, settings

import coordinax.embeddings as cxe
import coordinax_hypothesis.core as cxst


@given(kwargs=cxst.chart_init_kwargs(cxe.EmbeddedManifold))
@settings(max_examples=5)
def test_embedded_manifold_kwargs(kwargs: dict) -> None:
    """Test chart_init_kwargs generates valid EmbeddedManifold kwargs."""
    assert "intrinsic_chart" in kwargs
    assert "ambient_chart" in kwargs
    assert "params" in kwargs
    chart = cxe.EmbeddedManifold(**kwargs)
    assert isinstance(chart, cxe.EmbeddedManifold)
