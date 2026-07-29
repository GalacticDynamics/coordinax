"""`coordinaxs.hypothesis.main` re-exports every submodule strategy by identity.

Replaces the `test_also_accessible_via_main` method that each strategy module
carried, which asserted one row of this table apiece -- and, unlike those,
fails if a *new* strategy is added to a submodule but never surfaced on `main`.
"""

__all__: tuple[str, ...] = ()

import pytest

import coordinaxs.hypothesis.angles as cxast
import coordinaxs.hypothesis.charts as cxcst
import coordinaxs.hypothesis.distances as cxdst
import coordinaxs.hypothesis.main as cxst
import coordinaxs.hypothesis.manifolds as cxmst
import coordinaxs.hypothesis.representations as cxrst

#: Submodules whose public strategies `main` aggregates.
SUBMODULES = (cxast, cxcst, cxdst, cxmst, cxrst)


def _owner_of(name: str):
    """The submodule that defines *name*, or None."""
    return next((m for m in SUBMODULES if name in m.__all__), None)


@pytest.mark.parametrize("name", sorted(cxst.__all__))
def test_main_reexport_is_the_submodule_object(name: str) -> None:
    """Each `main` export is the very object its submodule defines."""
    owner = _owner_of(name)
    assert owner is not None, f"main exports {name!r}, which no submodule owns"
    assert getattr(cxst, name) is getattr(owner, name)
