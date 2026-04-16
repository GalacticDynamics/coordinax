"""Tests for spacetime charts (spacetime.py).

SpaceTimeCT: flat-key product chart for spacetime coordinates.
"""

import coordinax.charts as cxc


class TestSpaceTimeCTFlatKeys:
    """Test SpaceTimeCT uses flat string keys."""

    def test_factor_names_is_none(self) -> None:
        """SpaceTimeCT.factor_names should be None (flat-key specialization)."""
        st = cxc.SpaceTimeCT(cxc.cart3d)
        assert st.factor_names == ("time", "space")

    def test_components_are_flat_strings(self) -> None:
        """SpaceTimeCT components should be flat strings like 'ct', 'x', etc."""
        st = cxc.SpaceTimeCT(cxc.cart3d)
        assert st.components == ("ct", "x", "y", "z")

    def test_split_components_preserves_ct(self) -> None:
        """split_components should keep 'ct' key for time factor."""
        st = cxc.SpaceTimeCT(cxc.cart3d)
        p = {"ct": 1.0, "x": 2.0, "y": 3.0, "z": 4.0}
        parts = st.split_components(p)
        assert parts[0] == {"ct": 1.0}
        assert parts[1] == {"x": 2.0, "y": 3.0, "z": 4.0}

    def test_merge_components_preserves_ct(self) -> None:
        """merge_components should preserve 'ct' key."""
        st = cxc.SpaceTimeCT(cxc.cart3d)
        parts = ({"ct": 1.0}, {"x": 2.0, "y": 3.0, "z": 4.0})
        merged = st.merge_components(parts)
        assert merged == {"ct": 1.0, "x": 2.0, "y": 3.0, "z": 4.0}

    def test_split_merge_roundtrip(self) -> None:
        """Split followed by merge should recover original dict."""
        st = cxc.SpaceTimeCT(cxc.cart3d)
        original = {"ct": 1.0, "x": 2.0, "y": 3.0, "z": 4.0}
        parts = st.split_components(original)
        recovered = st.merge_components(parts)
        assert recovered == original
