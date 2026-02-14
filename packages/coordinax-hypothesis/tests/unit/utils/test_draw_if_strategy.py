"""Tests for ``coordinax_hypothesis.utils.draw_if_strategy``."""

import hypothesis.strategies as st
from hypothesis import given

from coordinax_hypothesis.utils import draw_if_strategy


class TestDrawIfStrategy:
    """Tests for draw_if_strategy."""

    @given(st.data())
    def test_returns_plain_value_unchanged(self, data: st.DataObject):
        """A non-strategy value is returned as-is."""
        value = 42
        result = draw_if_strategy(data.draw, value)
        assert result is value

    @given(st.data())
    def test_returns_plain_none_unchanged(self, data: st.DataObject):
        """None (a non-strategy) is returned as-is."""
        result = draw_if_strategy(data.draw, None)
        assert result is None

    @given(st.data())
    def test_returns_plain_string_unchanged(self, data: st.DataObject):
        """A plain string is returned without drawing."""
        value = "hello"
        result = draw_if_strategy(data.draw, value)
        assert result is value

    @given(st.data())
    def test_draws_from_strategy(self, data: st.DataObject):
        """When given a SearchStrategy, it draws a value from it."""
        strategy = st.integers(min_value=0, max_value=100)
        result = draw_if_strategy(data.draw, strategy)
        assert isinstance(result, int)
        assert 0 <= result <= 100

    @given(st.data())
    def test_draws_from_just_strategy(self, data: st.DataObject):
        """st.just(x) is a strategy, so draw_if_strategy should draw from it."""
        sentinel = object()
        result = draw_if_strategy(data.draw, st.just(sentinel))
        assert result is sentinel

    @given(st.data())
    def test_draws_from_sampled_from(self, data: st.DataObject):
        """Works with st.sampled_from."""
        options = ("a", "b", "c")
        result = draw_if_strategy(data.draw, st.sampled_from(options))
        assert result in options

    @given(st.data())
    def test_plain_list_not_drawn(self, data: st.DataObject):
        """A list (non-strategy) is returned as-is, not iterated."""
        value = [1, 2, 3]
        result = draw_if_strategy(data.draw, value)
        assert result is value

    @given(st.data())
    def test_plain_dict_not_drawn(self, data: st.DataObject):
        """A dict (non-strategy) is returned as-is."""
        value = {"key": "val"}
        result = draw_if_strategy(data.draw, value)
        assert result is value
