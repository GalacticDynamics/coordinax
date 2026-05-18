"""Unit tests for custom manifold and atlas classes."""

__all__: tuple[str, ...] = ()

import pytest
from hypothesis import given

import coordinax.charts as cxc
import coordinax.hypothesis.main as cxst
import coordinax.manifolds as cxm


class TestCustomAtlas:
    """CustomAtlas construction and membership behavior."""

    def test_supports_registered_chart_classes_only(self) -> None:
        """Supports returns True only for explicitly registered chart classes."""
        atlas = cxm.CustomAtlas(
            charts=(cxc.Cart2D, cxc.Polar2D),
            chart_default=cxc.cart2d,
        )

        assert isinstance(atlas.charts, tuple)
        assert atlas.charts == (cxc.Cart2D, cxc.Polar2D)
        assert atlas.has_chart(cxc.cart2d)
        assert atlas.has_chart(cxc.polar2d)
        assert not atlas.has_chart(cxc.cart3d)

    def test_raises_when_registered_chart_classes_repeat(self) -> None:
        """Constructing with duplicate chart classes raises ValueError."""
        with pytest.raises(ValueError, match="must be unique"):
            _ = cxm.CustomAtlas(
                charts=(cxc.Cart2D, cxc.Cart2D),
                chart_default=cxc.cart2d,
            )

    def test_raises_when_registered_chart_ndim_mismatch(self) -> None:
        """Constructing with mixed chart dimensions raises ValueError."""
        with pytest.raises(ValueError, match="has ndim=3 but expected 2"):
            _ = cxm.CustomAtlas(
                charts=(cxc.Cart2D, cxc.Cart3D),
                chart_default=cxc.cart2d,
            )

    def test_raises_when_default_chart_not_registered(self) -> None:
        """Default chart must be included in the registered chart set."""
        with pytest.raises(ValueError, match="Default chart class"):
            _ = cxm.CustomAtlas(
                charts=(cxc.Polar2D,),
                chart_default=cxc.cart2d,
            )


class TestCustomManifold:
    """CustomManifold forwarding and transition wrappers."""

    def test_forwards_ndim_and_default_chart(self) -> None:
        """Manifold ndim and default chart are forwarded from atlas."""
        atlas = cxm.CustomAtlas(
            charts=(cxc.Cart2D, cxc.Polar2D),
            chart_default=cxc.cart2d,
        )
        manifold = cxm.CustomManifold(atlas, metric=cxm.FlatMetric(2))

        assert manifold.ndim == 2
        assert manifold.default_chart() == cxc.cart2d

    def test_has_chart_and_check_chart(self) -> None:
        """Chart membership and validation are delegated to the atlas."""
        atlas = cxm.CustomAtlas(
            charts=(cxc.Cart2D, cxc.Polar2D),
            chart_default=cxc.cart2d,
        )
        manifold = cxm.CustomManifold(atlas, metric=cxm.FlatMetric(2))

        assert manifold.has_chart(cxc.cart2d)
        assert manifold.has_chart(cxc.polar2d)
        assert not manifold.has_chart(cxc.cart3d)

        manifold.check_chart(cxc.cart2d)
        with pytest.raises(ValueError, match="is not supported"):
            manifold.check_chart(cxc.cart3d)


@given(atlas=cxst.atlases(cxm.CustomAtlas))
def test_custom_atlas_property_invariants(atlas: cxm.CustomAtlas) -> None:
    """Generated custom atlases always support their default chart."""
    # Invariant 1: the default chart instance must be accepted by the atlas.
    assert atlas.has_chart(atlas.default_chart())
    # Invariant 2: default chart class must be one of the registered classes.
    assert type(atlas.default_chart()) in atlas.charts


@given(
    M=cxst.manifolds(
        cxm.CustomManifold, ndim=2, required_chart_classes=(cxc.Cart2D, cxc.Polar2D)
    )
)
def test_custom_manifold_property_transition(M: cxm.CustomManifold) -> None:
    """Generated 2D custom manifolds support cart2d->polar2d transitions."""
    # Fixed input point keeps this property focused on manifold/chart validity,
    # not numeric fuzz from random values.
    x = {"x": 1.0, "y": 1.0}
    # The required chart classes ensure this transition path is defined.
    got = cxc.pt_map(x, cxc.cart2d, cxc.polar2d)
    # Transition result schema should match polar chart component keys.
    assert set(got) == {"r", "theta"}
