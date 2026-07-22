"""Packaging metadata checks for coordinaxs.hypothesis."""

__all__: tuple[str, ...] = ()

import pathlib

import coordinaxs.hypothesis.main as cxst

# ``coordinaxs.hypothesis`` is a namespace package split across distributions,
# so anchor the check on ``main`` (only this distribution provides it) and
# resolve the enclosing ``coordinaxs/hypothesis`` directory. ``__file__`` is
# ``.../hypothesis/main/__init__.py`` when ``main`` is a package (→ parents[1])
# or ``.../hypothesis/main.py`` when it is a module (→ parents[0]); handle both
# so the check is robust to the layout.
_MAIN = pathlib.Path(cxst.__file__).resolve()
_PORTION = _MAIN.parents[1] if _MAIN.name == "__init__.py" else _MAIN.parents[0]


def test_ships_py_typed_marker() -> None:
    """The package declares ``Typing :: Typed`` so it must ship ``py.typed``.

    The check is scoped to *this* distribution's own namespace portion, not the
    shared ``coordinaxs.hypothesis`` namespace as a whole.
    """
    marker = _PORTION / "py.typed"
    assert marker.is_file(), f"py.typed marker missing from {_PORTION}"
