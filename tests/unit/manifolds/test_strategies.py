"""Property tests using general manifold strategies."""

__all__: tuple[str, ...] = ()

from hypothesis import given

import coordinax.charts as cxc
import coordinax.hypothesis.main as cxst
import coordinax.manifolds as cxm


@given(atlas=cxst.atlases())
def test_generated_atlas_supports_default_chart(atlas: cxm.AbstractAtlas) -> None:
    """Every generated atlas exposes a chart object as its default chart."""
    assert isinstance(atlas.default_chart(), cxc.AbstractChart)


@given(manifold=cxst.manifolds())
def test_generated_manifold_supports_default_chart(
    manifold: cxm.AbstractManifold,
) -> None:
    """Every generated manifold exposes an atlas and a default chart."""
    assert isinstance(manifold.atlas, cxm.AbstractAtlas)
    assert manifold.atlas.ndim == manifold.ndim
    assert isinstance(manifold.default_chart(), cxc.AbstractChart)
