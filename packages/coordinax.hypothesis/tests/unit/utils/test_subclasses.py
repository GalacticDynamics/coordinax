"""Tests for ``coordinax.hypothesis.utils._src.subclasses``."""

import sys
import types

import coordinax.charts as cxc

from coordinax.hypothesis.utils._src.subclasses import canonicalize_coordinax_class


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
    setattr(synthetic_mod, "SyntheticThing", canonical_cls)
    monkeypatch.setitem(sys.modules, "coordinax.synthetic", synthetic_mod)

    duplicate_cls = type(
        "SyntheticThing",
        (),
        {"__module__": "coordinax._src.synthetic.deep"},
    )

    assert canonicalize_coordinax_class(duplicate_cls) is canonical_cls


def test_canonicalize_real_chart_class_returns_public_class() -> None:
    """Real chart classes are canonicalized to the public class object."""

    canonicalize_coordinax_class.cache_clear()

    assert canonicalize_coordinax_class(cxc.Cart3D) is cxc.Cart3D
