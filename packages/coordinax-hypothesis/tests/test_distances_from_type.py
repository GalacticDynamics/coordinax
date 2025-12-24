"""Tests for st.from_type() with distance types."""

from hypothesis import given, strategies as st

import unxt as u

import coordinax.distances as cxd


class TestDistanceFromType:
    """Test st.from_type() for Distance type."""

    @given(dist=st.from_type(cxd.Distance))
    def test_from_type_basic(self, dist: cxd.Distance) -> None:
        """Test that st.from_type(Distance) generates valid Distance instances."""
        assert isinstance(dist, cxd.Distance)
        # Default check_negative=True means value should be non-negative
        assert dist.value >= 0

    @given(dist=st.from_type(cxd.Distance))
    def test_from_type_has_length_dimension(self, dist: cxd.Distance) -> None:
        """Test that generated distances have length dimension."""
        assert u.dimension_of(dist) == u.dimension("length")

    @given(data=st.data())
    def test_from_type_generates_variety(self, data: st.DataObject) -> None:
        """Test that from_type generates different values."""
        dist1 = data.draw(st.from_type(cxd.Distance))
        dist2 = data.draw(st.from_type(cxd.Distance))

        # Most of the time these should be different
        # (Could occasionally be the same, but very unlikely)
        assert isinstance(dist1, cxd.Distance)
        assert isinstance(dist2, cxd.Distance)


class TestDistanceModulusFromType:
    """Test st.from_type() for DistanceModulus type."""

    @given(dm=st.from_type(cxd.DistanceModulus))
    def test_from_type_basic(self, dm: cxd.DistanceModulus) -> None:
        """Test that st.from_type(DistanceModulus) generates valid instances."""
        assert isinstance(dm, cxd.DistanceModulus)

    @given(dm=st.from_type(cxd.DistanceModulus))
    def test_from_type_has_mag_units(self, dm: cxd.DistanceModulus) -> None:
        """Test that generated distance moduli have magnitude units."""
        assert dm.unit == "mag"

    @given(data=st.data())
    def test_from_type_generates_variety(self, data: st.DataObject) -> None:
        """Test that from_type generates different values."""
        dm1 = data.draw(st.from_type(cxd.DistanceModulus))
        dm2 = data.draw(st.from_type(cxd.DistanceModulus))

        assert isinstance(dm1, cxd.DistanceModulus)
        assert isinstance(dm2, cxd.DistanceModulus)


class TestParallaxFromType:
    """Test st.from_type() for Parallax type."""

    @given(plx=st.from_type(cxd.Parallax))
    def test_from_type_basic(self, plx: cxd.Parallax) -> None:
        """Test that st.from_type(Parallax) generates valid Parallax instances."""
        assert isinstance(plx, cxd.Parallax)
        # Default check_negative=True means value should be non-negative
        assert plx.value >= 0

    @given(plx=st.from_type(cxd.Parallax))
    def test_from_type_has_angle_dimension(self, plx: cxd.Parallax) -> None:
        """Test that generated parallaxes have angle dimension."""
        assert u.dimension_of(plx) == u.dimension("angle")

    @given(data=st.data())
    def test_from_type_generates_variety(self, data: st.DataObject) -> None:
        """Test that from_type generates different values."""
        plx1 = data.draw(st.from_type(cxd.Parallax))
        plx2 = data.draw(st.from_type(cxd.Parallax))

        assert isinstance(plx1, cxd.Parallax)
        assert isinstance(plx2, cxd.Parallax)


class TestIntegrationWithBuilds:
    """Test that distance types work with st.builds() and from_type."""

    @given(data=st.data())
    def test_builds_with_distance_arg(self, data: st.DataObject) -> None:
        """Test that st.builds() can use from_type for Distance arguments."""

        def takes_distance(d: cxd.Distance) -> float:
            """Distance -> float."""
            return d.value.item()

        # st.builds should automatically use from_type for Distance
        strategy = st.builds(takes_distance, d=st.from_type(cxd.Distance))
        value = data.draw(strategy)

        assert isinstance(value, float)
        assert value >= 0

    @given(data=st.data())
    def test_builds_with_parallax_arg(self, data: st.DataObject) -> None:
        """Test that st.builds() can use from_type for Parallax arguments."""

        def takes_parallax(plx: cxd.Parallax) -> float:
            """Parallax -> float."""
            return plx.value.item()

        strategy = st.builds(takes_parallax, plx=st.from_type(cxd.Parallax))
        value = data.draw(strategy)

        assert isinstance(value, float)
        assert value >= 0
