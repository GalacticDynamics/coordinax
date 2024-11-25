"""Test :mod:`coordinax.utils`."""

from warnings import warn

import pytest

from coordinax._src.vectors.exceptions import IrreversibleDimensionChange


def test_warning_irreversibledimensionchange():
    """Test :class:`IrreversibleDimensionChange`."""
    with pytest.warns(UserWarning, match="Irreversible dimension change"):
        warn("Irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
