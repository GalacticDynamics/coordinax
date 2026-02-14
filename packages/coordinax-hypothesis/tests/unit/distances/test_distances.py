"""Tests for the distances strategy."""

import hypothesis.strategies as st
from hypothesis import given, settings

import unxt as u

import coordinax.distances as cxd
import coordinax_hypothesis.core as cxst


@given(dist=cxst.distances())
@settings(max_examples=50)
def test_basic_distance(dist: cxd.Distance) -> None:
    """Test basic distance generation."""
    assert isinstance(dist, cxd.Distance)
    assert dist.shape == ()
    assert dist.value >= 0  # default check_negative=True


@given(dist=cxst.distances(check_negative=False))
@settings(max_examples=50)
def test_distance_allow_negative(dist: cxd.Distance) -> None:
    """Test distance generation with negative values allowed."""
    assert isinstance(dist, cxd.Distance)
    # Don't check sign when check_negative=False


@given(dist=cxst.distances(unit="kpc"))
@settings(max_examples=50)
def test_distance_with_units(dist: cxd.Distance) -> None:
    """Test distance generation with specific units."""
    assert isinstance(dist, cxd.Distance)
    assert dist.unit == "kpc"


@given(dist=cxst.distances(shape=5))
@settings(max_examples=30)
def test_distance_vector(dist: cxd.Distance) -> None:
    """Test vector distance generation."""
    assert isinstance(dist, cxd.Distance)
    assert dist.shape == (5,)
    assert all(dist.value >= 0)  # all elements should be non-negative


@given(dist=cxst.distances(shape=(2, 3)))
@settings(max_examples=30)
def test_distance_2d(dist: cxd.Distance) -> None:
    """Test 2D distance array generation."""
    assert isinstance(dist, cxd.Distance)
    assert dist.shape == (2, 3)


@given(dist=cxst.distances(check_negative=st.sampled_from([True, False])))
@settings(max_examples=50)
def test_distance_with_strategy_check_negative(dist: cxd.Distance) -> None:
    """Test distance with check_negative as a strategy."""
    assert isinstance(dist, cxd.Distance)
    # check_negative varies, so we can't assert about the sign


@given(
    dist=cxst.distances(elements=st.floats(min_value=1.0, max_value=100.0, width=32))
)
@settings(max_examples=30)
def test_distance_with_custom_elements(dist: cxd.Distance) -> None:
    """Test distance with custom elements range."""
    assert isinstance(dist, cxd.Distance)
    assert 1.0 <= dist.value <= 100.0


@given(
    dist=cxst.distances(
        check_negative=True, elements=st.floats(min_value=0.0, max_value=10.0, width=32)
    )
)
@settings(max_examples=30)
def test_distance_check_negative_with_elements(dist: cxd.Distance) -> None:
    """Test that check_negative works with custom elements."""
    assert isinstance(dist, cxd.Distance)
    # When check_negative=True and elements provided, min_value should be adjusted
    assert dist.value >= 0
    assert dist.value <= 50.0


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
