"""Packaging metadata checks for coordinaxs.hypothesis."""

__all__: tuple[str, ...] = ()

import pathlib

import coordinaxs.hypothesis.main as cxst


def _distribution_portion_dir() -> pathlib.Path:
    """Return this distribution's ``coordinaxs/hypothesis`` directory.

    ``coordinaxs.hypothesis`` is a namespace package split across distributions,
    so anchor on ``main`` (only this distribution provides it) and resolve the
    enclosing directory. ``__file__`` is ``.../hypothesis/main/__init__.py`` for
    a package (→ parents[1]) or ``.../hypothesis/main.py`` for a module
    (→ parents[0]); handle both so the check is robust to the layout.
    """
    main_path = pathlib.Path(cxst.__file__).resolve()
    if main_path.name == "__init__.py":
        return main_path.parents[1]
    return main_path.parents[0]


def test_ships_py_typed_marker() -> None:
    """The package declares ``Typing :: Typed`` so it must ship ``py.typed``.

    The check is scoped to *this* distribution's own namespace portion, not the
    shared ``coordinaxs.hypothesis`` namespace as a whole.
    """
    portion_dir = _distribution_portion_dir()
    marker = portion_dir / "py.typed"
    assert marker.is_file(), f"py.typed marker missing from {portion_dir}"
