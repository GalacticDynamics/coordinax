"""Tests for ``coordinaxs.hypothesis.utils._src.subclasses``."""

import subprocess
import sys

import types

import coordinax.charts as cxc

from coordinaxs.hypothesis.utils._src.subclasses import (
    canonicalize_coordinax_class,
    is_test_declared,
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


#: Declares a chart outside coordinax and asks whether the walk returns it,
#: with the cache warmed before it exists and again cleared after.
#:
#: Out-of-process, and not merely to isolate the cache: subclassing a real chart
#: registers it with coordinax's own plum dispatch, where `guess_chart` will
#: pick it for ``{x, y, z}`` and fail every later test that builds a point from
#: a bare dict. Declaring it here would break this suite exactly the way the
#: provenance rule exists to prevent.
_CALLER_DECLARED = """
import warnings
warnings.filterwarnings("ignore")
import coordinax.charts as cxc
from coordinaxs.hypothesis.utils import get_all_subclasses as g

warm = g(cxc.AbstractChart, exclude_abstract=True)
class LocalChart(cxc.Cart3D): pass          # a caller's class, not coordinax's
print(int(LocalChart in g(cxc.AbstractChart, exclude_abstract=True)))
g.cache_clear()
cold = g(cxc.AbstractChart, exclude_abstract=True)
print(int(LocalChart in cold))
print(int(set(cold) == set(warm)))
"""


def test_caller_declared_subclasses_are_not_returned() -> None:
    """A class declared outside coordinax never enters the candidate set.

    Order-independent, unlike the cache: checked both with the cache warmed
    before the class exists and with it cleared afterwards, since that ordering
    used to decide the answer.
    """
    proc = subprocess.run(  # noqa: S603  # fixed literal script, this interpreter
        [sys.executable, "-c", _CALLER_DECLARED],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    in_warm, in_cold, library_unchanged = proc.stdout.split()
    assert in_warm == "0", "a caller's class reached a warm cache"
    assert in_cold == "0", "a caller's class reached a cold cache"
    assert library_unchanged == "1", "library classes changed"


def test_test_provenance_is_recognised() -> None:
    """Test modules are spotted; ordinary library modules are not.

    The downstream case is the point: a package that defines its own charts is a
    first-class user of these strategies, so its classes must survive.
    """
    assert is_test_declared(test_test_provenance_is_recognised.__class__) is False
    assert is_test_declared(cxc.Cart3D) is False  # coordinax itself
    for module in (
        "tests.unit.test_x",
        "mypkg.tests.helpers",
        "pkg.conftest",
        "__main__",
    ):
        fake = type("Fake", (), {"__module__": module})
        assert is_test_declared(fake), module
    for module in ("mypkg", "mypkg.charts", "coordinaxs.astro._src.frames"):
        real = type("Real", (), {"__module__": module})
        assert not is_test_declared(real), module
