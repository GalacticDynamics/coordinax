"""Test :mod:`vector._utils`."""

from warnings import warn

import pytest

from vector._exceptions import IrreversibleDimensionChange


def test_warning_irreversibledimensionchange():
    """Test :class:`IrreversibleDimensionChange`."""
    with pytest.warns(UserWarning, match="Irreversible dimension change"):
        warn("Irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
