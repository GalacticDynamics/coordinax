"""Regression tests for canonical module import identities during pytest runs."""

import importlib
import pathlib
import sys

from types import ModuleType

import coordinax.charts as cxc
from coordinax.charts._src.base import AbstractChart

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _is_workspace_module(module: ModuleType, /) -> bool:
    """Return ``True`` when a module file lives in the coordinax workspace."""
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    return pathlib.Path(module_file).resolve().is_relative_to(WORKSPACE_ROOT)


def test_canonical_chart_module_identity() -> None:
    """Chart classes resolve from canonical ``coordinax.*`` module paths."""
    module = importlib.import_module("coordinax.charts._src.d3")
    assert module.Cart3D is cxc.Cart3D
    assert cxc.Cart3D.__module__.startswith("coordinax.charts.")


def test_workspace_short_alias_modules_are_not_loaded() -> None:
    """Workspace chart modules are never loaded under short ``charts.*`` aliases."""
    aliases = [
        name
        for name, module in sys.modules.items()
        if name.startswith("charts") and _is_workspace_module(module)
    ]
    assert aliases == []


def test_canonical_chart_is_abstractchart_instance() -> None:
    """Canonical chart instances satisfy the abstract chart contract."""
    assert isinstance(cxc.cart3d, AbstractChart)
