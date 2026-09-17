"""Every inter-package dependency must carry a version floor.

Workspace deps (``coordinax`` / ``coordinaxs.*``) are sourced locally via
``[tool.uv.sources]`` during development, but that key is stripped from
published metadata. Without an explicit floor a released package could resolve
against an incompatible sibling on PyPI, so guard the floors here.
"""

__all__: tuple[str, ...] = ()

import pathlib
import re
import tomllib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PYPROJECTS = [
    _ROOT / "pyproject.toml",
    *sorted((_ROOT / "packages").glob("*/pyproject.toml")),
]


def _requirement_name(req: str) -> str:
    """The distribution name at the start of a PEP 508 requirement string.

    PEP 508 names are case-insensitive, so the name is case-folded for robust
    matching. (Only case is folded, not the ``.``/``-`` separators, so the
    ``coordinaxs.`` namespace prefix stays intact.)
    """
    match = re.match(r"\s*([A-Za-z0-9._-]+)", req)
    return match.group(1).casefold() if match else ""


def _interpackage_reqs(cfg: dict) -> list[str]:
    """All ``coordinax``/``coordinaxs.*`` requirement strings in a pyproject.

    Selection is by the parsed requirement *name*, so every specifier form
    (``>=``, ``<``, ``!=``, ``~=``, extras, markers) is covered.
    """
    proj = cfg.get("project", {})
    reqs: list[str] = list(proj.get("dependencies", []))
    for extra in proj.get("optional-dependencies", {}).values():
        reqs += extra
    for group in cfg.get("dependency-groups", {}).values():
        reqs += [g for g in group if isinstance(g, str)]
    return [
        r
        for r in reqs
        if (name := _requirement_name(r)) == "coordinax"
        or name.startswith("coordinaxs.")
    ]


@pytest.mark.parametrize("path", _PYPROJECTS, ids=lambda p: p.parent.name)
def test_interpackage_deps_have_floor(path: pathlib.Path) -> None:
    """Each coordinax/coordinaxs.* requirement declares a ``>=`` floor."""
    cfg = tomllib.loads(path.read_text())
    # Strip any environment marker (which may itself contain ``>=``) before
    # inspecting the version specifier.
    unpinned = [r for r in _interpackage_reqs(cfg) if ">=" not in r.split(";", 1)[0]]
    assert not unpinned, f"{path.parent.name}: unpinned inter-package deps: {unpinned}"
