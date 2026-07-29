"""`coordinaxs.hypothesis.main` aggregates the submodule strategies.

Replaces the `test_also_accessible_via_main` method that each strategy module
carried, which asserted one row of this apiece and so could not notice a
submodule export that never reached `main`.

Both directions are checked:

* everything `main` exports is the same object its submodule defines, and
* everything the submodules export reaches `main`, bar a documented exception.
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

#: Submodule exports `main` deliberately does not re-export.
#:
#: `make_nonnegative` is an element-strategy helper, not a strategy in its own
#: right; it is used to *build* `distances`, so it stays on the submodule.
NOT_ON_MAIN = frozenset({"make_nonnegative"})


def _owners_of(name: str) -> list:
    """Every submodule exporting *name*.

    A list, not a single module: `cdicts` is exported by both `charts` and
    `representations`. They are currently the same object, and the identity
    assertion below is what would catch them drifting apart.
    """
    return [mod for mod in SUBMODULES if name in mod.__all__]


@pytest.mark.parametrize("name", sorted(cxst.__all__))
def test_main_reexport_is_the_submodule_object(name: str) -> None:
    """Each `main` export is the very object its submodule(s) define."""
    owners = _owners_of(name)
    assert owners, f"main exports {name!r}, which no submodule owns"
    for owner in owners:
        assert getattr(cxst, name) is getattr(owner, name), (
            f"main.{name} is not {owner.__name__}.{name}"
        )


SUBMODULE_EXPORTS = sorted(
    (mod.__name__.rsplit(".", 1)[-1], name)
    for mod in SUBMODULES
    for name in mod.__all__
    if name not in NOT_ON_MAIN
)


@pytest.mark.parametrize(
    ("module_name", "name"),
    SUBMODULE_EXPORTS,
    ids=[f"{m}.{n}" for m, n in SUBMODULE_EXPORTS],
)
def test_submodule_export_reaches_main(module_name: str, name: str) -> None:
    """Adding a strategy to a submodule but not to `main` is a failure here."""
    assert hasattr(cxst, name), (
        f"{module_name}.{name} is not re-exported from main; "
        f"add it to main, or to NOT_ON_MAIN with a reason"
    )


def test_not_on_main_entries_are_still_real_exports() -> None:
    """`NOT_ON_MAIN` cannot silently rot into naming something that is gone."""
    all_owned = {name for mod in SUBMODULES for name in mod.__all__}
    assert all_owned >= NOT_ON_MAIN
    assert not (NOT_ON_MAIN & set(cxst.__all__))
