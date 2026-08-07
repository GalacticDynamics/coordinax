"""Tests for ``coordinaxs.hypothesis.utils._src.subclasses``."""

import subprocess
import sys

import types

import coordinax.charts as cxc

from coordinaxs.hypothesis.utils._src.subclasses import canonicalize_coordinax_class


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


#: Asks a fresh interpreter what `get_all_subclasses` returns for a subclass
#: declared before vs after the cache is warmed.
#:
#: Out-of-process because declaring the subclass here would leave it in
#: ``__subclasses__`` for the rest of the session, where the strategies would
#: hand it out as though it were a library type.
_CACHE_ORDER = """
import warnings
warnings.filterwarnings("ignore")
import coordinax.representations as cxr
from coordinaxs.hypothesis.utils import get_all_subclasses as g

base = cxr.AbstractBasis
warm = len(g(base, exclude_abstract=True))          # warm the cache first
class LateBasis(cxr.AbstractLinearBasis): pass      # ... then declare
print(len(g(base, exclude_abstract=True)) - warm)   # 0: absent

g.cache_clear()                                     # cold cache, class exists
print(len(g(base, exclude_abstract=True)) - warm)   # 1: present
"""


def test_result_depends_on_when_the_cache_was_warmed() -> None:
    """The cached candidate set is order-dependent, as the docstring warns.

    Pinned so the warning cannot quietly stop being true. If someone makes the
    candidate set import-time — the fix the docstring points at — this fails and
    says so, rather than the warning rotting into fiction.
    """
    proc = subprocess.run(  # noqa: S603  # fixed literal script, this interpreter
        [sys.executable, "-c", _CACHE_ORDER],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    declared_after_warming, declared_before_warming = proc.stdout.split()
    assert declared_after_warming == "0", "a later subclass leaked into a warm cache"
    assert declared_before_warming == "1", "a cold cache missed an existing subclass"
