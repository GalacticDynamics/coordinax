"""Packaging metadata checks for coordinaxs.hypothesis."""

__all__: tuple[str, ...] = ()

import pathlib

import coordinaxs.hypothesis.main as cxst


def test_ships_py_typed_marker() -> None:
    """The package declares ``Typing :: Typed`` so it must ship ``py.typed``.

    ``coordinaxs.hypothesis`` is a namespace package split across two
    distributions, so anchor the check to *this* distribution's portion (the
    one containing the ``main`` subpackage) rather than the namespace as a whole.
    """
    portion = pathlib.Path(cxst.__file__).parent.parent  # coordinaxs/hypothesis
    assert (portion / "py.typed").is_file()
