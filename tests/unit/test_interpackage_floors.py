"""Every inter-package dependency must carry a version floor.

Workspace deps (``coordinax`` / ``coordinaxs.*``) are sourced locally via
``[tool.uv.sources]`` during development, but that key is stripped from
published metadata. Without an explicit floor a released package could resolve
against an incompatible sibling on PyPI, so guard the floors here.
"""

__all__: tuple[str, ...] = ()

import pathlib
import tomllib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PYPROJECTS = [
    _ROOT / "pyproject.toml",
    *sorted((_ROOT / "packages").glob("*/pyproject.toml")),
]


def _interpackage_reqs(cfg: dict) -> list[str]:
    """All ``coordinax``/``coordinaxs.*`` requirement strings in a pyproject."""
    proj = cfg.get("project", {})
    reqs: list[str] = list(proj.get("dependencies", []))
    for extra in proj.get("optional-dependencies", {}).values():
        reqs += extra
    for group in cfg.get("dependency-groups", {}).values():
        reqs += [g for g in group if isinstance(g, str)]
    return [
        r
        for r in reqs
        if r == "coordinax"
        or r.startswith(
            ("coordinax>", "coordinax=", "coordinax[", "coordinax ", "coordinaxs.")
        )
    ]


@pytest.mark.parametrize("path", _PYPROJECTS, ids=lambda p: p.parent.name)
def test_interpackage_deps_have_floor(path: pathlib.Path) -> None:
    """Each coordinax/coordinaxs.* requirement declares a ``>=`` floor."""
    cfg = tomllib.loads(path.read_text())
    unpinned = [r for r in _interpackage_reqs(cfg) if ">=" not in r]
    assert not unpinned, f"{path.parent.name}: unpinned inter-package deps: {unpinned}"
