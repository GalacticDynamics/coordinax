"""Tests for ``coordinaxs.hypothesis.utils._src.subclasses``."""

import sys

import types

import coordinax.charts as cxc

from coordinaxs.hypothesis.utils._src.subclasses import (
    _public_coordinax_module_candidates,
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


def test_canonicalize_falls_through_when_the_qualname_is_absent() -> None:
    """A coordinax class no candidate module exposes comes back unchanged.

    Exercises the `AttributeError -> continue` arm and the final `return cls`.
    Those turn on what `sys.modules` happens to hold, which varies with import
    order and xdist worker, so without a test they are covered by luck.
    """
    canonicalize_coordinax_class.cache_clear()

    orphan = type("NotExportedAnywhere", (), {"__module__": "coordinax._src.nowhere"})

    assert canonicalize_coordinax_class(orphan) is orphan


def test_canonicalize_ignores_a_non_class_attribute(monkeypatch) -> None:
    """`isinstance(resolved, type)` is False, so the candidate is not taken."""
    canonicalize_coordinax_class.cache_clear()

    shadow = types.ModuleType("coordinax.shadow")
    shadow.Decoy = "not a class"  # same name, wrong kind
    monkeypatch.setitem(sys.modules, "coordinax.shadow", shadow)

    cls = type("Decoy", (), {"__module__": "coordinax._src.shadow"})

    assert canonicalize_coordinax_class(cls) is cls


def test_module_candidates_walk_up_and_stop_at_coordinax() -> None:
    """The parent walk ends at `coordinax`, and never above it."""
    got = _public_coordinax_module_candidates("coordinax._src.charts.deep.inner")

    assert got[0] == "coordinax.charts.deep.inner"
    assert "coordinax" in got
    assert all("." in c or c == "coordinax" for c in got)


def test_module_candidates_are_deduplicated() -> None:
    """`dict.fromkeys` is doing real work: loaded modules repeat the parents."""
    got = _public_coordinax_module_candidates("coordinax._src.charts")

    assert len(got) == len(set(got))
