"""Tests for ``coordinaxs.hypothesis.utils._src.subclasses``."""

import sys

import types

import coordinax.charts as cxc

from coordinaxs.hypothesis.utils._src.subclasses import (
    canonicalize_coordinax_class,
    get_all_subclasses,
)


def test_canonicalize_non_coordinax_class_is_identity() -> None:
    """Non-coordinax classes are returned unchanged."""

    class _Plain:
        pass

    assert canonicalize_coordinax_class(_Plain) is _Plain


def test_canonicalize_resolves_via_public_parent_module(monkeypatch) -> None:
    """Canonicalization resolves through public parent modules dynamically."""
    canonicalize_coordinax_class.cache_clear()

    synthetic_mod = types.ModuleType("coordinax.synthetic")
    canonical_cls: type = type(
        "SyntheticThing", (), {"__module__": "coordinax.synthetic"}
    )
    synthetic_mod.SyntheticThing = canonical_cls
    monkeypatch.setitem(sys.modules, "coordinax.synthetic", synthetic_mod)

    duplicate_cls = type(
        "SyntheticThing", (), {"__module__": "coordinax._src.synthetic.deep"}
    )

    assert canonicalize_coordinax_class(duplicate_cls) is canonical_cls


def test_canonicalize_real_chart_class_returns_public_class() -> None:
    """Real chart classes are canonicalized to the public class object."""
    canonicalize_coordinax_class.cache_clear()

    assert canonicalize_coordinax_class(cxc.Cart3D) is cxc.Cart3D


class _GuardBase:
    """Root of a throwaway hierarchy, kept off the chart classes on purpose."""


class _GuardChild(_GuardBase):
    """A module-level subclass, i.e. the kind that should still be drawn."""


def test_classes_defined_inside_a_function_are_not_drawn() -> None:
    """A class defined in a function body is never a library class (#866).

    ``__subclasses__`` keeps reporting a class whose ``__init_subclass__``
    raised, for as long as a traceback holds it alive, and a throwaway defined
    in a test would otherwise leak into every later draw.

    Deliberately not a chart subclass: ``AbstractChart.__init_subclass__``
    registers one in ``NON_ABC_CHART_CLASSES``, which `guess_chart` reads, so
    a local chart here would make ``('x', 'y', 'z')`` ambiguous for the rest of
    the session -- which is the very bug under test.
    """

    class _Local(_GuardBase):
        """Defined inside the function, so it must not be drawn."""

    get_all_subclasses.cache_clear()
    result = get_all_subclasses(_GuardBase)

    assert _GuardChild in result
    assert _Local not in result
