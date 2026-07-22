"""Packaging metadata checks for coordinaxs.hypothesis."""

__all__: tuple[str, ...] = ()

import pathlib

import pytest

import coordinaxs.hypothesis.main as cxst


def _portion_dir_for(main_file: str) -> pathlib.Path:
    """Resolve the ``coordinaxs/hypothesis`` directory from ``main``'s file.

    ``coordinaxs.hypothesis`` is a namespace package split across distributions,
    so anchor on ``main`` (only this distribution provides it). ``main`` may be
    a package (``.../hypothesis/main/__init__.py`` → ``.parent.parent``) or a
    module (``.../hypothesis/main.py`` → ``.parent``); handle both. Compiled
    forms — a plain ``__init__.pyc`` or a ``__pycache__/name.cpython-XY.pyc``
    cache file — are normalised back to their source location first.
    """
    main_path = pathlib.Path(main_file)
    if main_path.parent.name == "__pycache__":
        # .../X/__pycache__/name.cpython-XY.pyc → .../X/name.py
        source_stem = main_path.name.split(".", 1)[0]
        main_path = main_path.parent.parent / f"{source_stem}.py"
    if main_path.stem == "__init__":
        return main_path.parent.parent
    return main_path.parent


def test_ships_py_typed_marker() -> None:
    """The package declares ``Typing :: Typed`` so it must ship ``py.typed``.

    The check is scoped to *this* distribution's own namespace portion, not the
    shared ``coordinaxs.hypothesis`` namespace as a whole.
    """
    portion_dir = _portion_dir_for(str(pathlib.Path(cxst.__file__).resolve()))
    marker = portion_dir / "py.typed"
    assert marker.is_file(), f"py.typed marker missing from {portion_dir}"


@pytest.mark.parametrize(
    "main_file",
    [
        "/pkg/src/coordinaxs/hypothesis/main/__init__.py",  # main as package
        "/pkg/src/coordinaxs/hypothesis/main/__init__.pyc",  # compiled package
        # cached compiled package (CPython __pycache__ layout)
        "/pkg/src/coordinaxs/hypothesis/main/__pycache__/__init__.cpython-312.pyc",
        "/pkg/src/coordinaxs/hypothesis/main.py",  # main as module
        # cached compiled module (CPython __pycache__ layout)
        "/pkg/src/coordinaxs/hypothesis/__pycache__/main.cpython-312.pyc",
    ],
)
def test_portion_dir_handles_both_layouts(main_file: str) -> None:
    """Both the package and module layouts resolve to ``coordinaxs/hypothesis``."""
    assert _portion_dir_for(main_file) == pathlib.Path("/pkg/src/coordinaxs/hypothesis")
