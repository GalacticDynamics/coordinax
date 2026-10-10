"""Test the `unxt` registrations in `coordinax`."""

from typing import Union

import pytest

import unxt as u

import coordinax._src.custom_types as ct
from coordinax._src.distances.custom_types import BBtLength


@pytest.mark.parametrize(
    "hint",
    [ct.BBtLength | BBtLength, Union[ct.BBtLength, BBtLength]],  # noqa: UP007
    ids=["UnionType", "typing.Union"],
)
def test_dimension_of_union(hint: object) -> None:
    """A union of same-dimension hints has that dimension."""
    assert u.dimension_of(hint) == u.dimension("length")
