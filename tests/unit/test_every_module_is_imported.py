"""Every module under ``src/coordinax`` must be reachable by importing it.

A module the package never imports is not merely dead -- it is dead *only in
production*. ``testpaths`` includes ``src/``, and collecting a file for its
doctests imports it, which runs its ``@plum.dispatch`` registrations. So an
unimported module's methods are absent when a user calls, and present when the
suite does, and the suite is the one place the divergence cannot be observed.

That is not hypothetical: `coordinax.vectors._src.register_manifolds` registers
``pt_project(Point, HyperSphericalManifold)``, and until it was wired up that
call raised `NotFoundLookupError` for users while its own doctests passed.
"""

__all__: tuple[str, ...] = ()

import sys
from pathlib import Path

from conftest import import_public_subpackages

import coordinax as cx

#: Modules that legitimately go unimported, with the reason.
ALLOWED_UNIMPORTED = {
    # Written by hatch-vcs at build time; nothing re-exports it, and it
    # registers nothing. See the packaging audit for whether it should exist.
    "coordinax._version"
}

_ROOT = Path(cx.__file__).parent


def _module_name(path: Path) -> str:
    """Dotted module name for a file under the `coordinax` package."""
    parts = path.relative_to(_ROOT.parent).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def test_no_module_is_dead_in_production_only() -> None:
    """No module under `src/coordinax` is skipped by the package's imports."""
    import_public_subpackages()
    imported = frozenset(sys.modules)
    unimported = sorted(
        name
        for path in _ROOT.rglob("*.py")
        if (name := _module_name(path)) not in imported
        and name not in ALLOWED_UNIMPORTED
    )
    assert not unimported, (
        "These modules are never imported by `coordinax`, so any dispatch they "
        "register exists only while the test suite is running: "
        f"{unimported}. Import them from the owning package's `__init__`, or "
        "delete them."
    )


def test_the_allowlist_has_no_stale_entries() -> None:
    """An allowlisted module that no longer exists should be dropped."""
    for name in ALLOWED_UNIMPORTED:
        base = _ROOT.parent / Path(*name.split("."))
        # `_module_name` collapses `pkg/__init__.py` to `pkg`, so an allowlisted
        # name may be either a module file or a package directory.
        assert base.with_suffix(".py").exists() or (base / "__init__.py").exists(), (
            f"{name} is allowlisted but no longer exists; drop the entry."
        )
