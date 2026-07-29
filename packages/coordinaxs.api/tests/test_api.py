"""Tests for the `coordinaxs.api` dispatch surface.

`coordinaxs.api` exists to declare the multiple-dispatch functions that the
implementation packages register against. There is nothing to exercise beyond
"the subpackage imports" and "the function has at least one method", so the
whole surface is one table.
"""

__all__: tuple[str, ...] = ()

import importlib

import pytest

#: Public dispatch functions exposed by each `coordinaxs.api` subpackage.
API_FUNCTIONS: dict[str, tuple[str, ...]] = {
    "charts": ("cartesian_chart", "pt_map", "guess_chart", "cdict"),
    "frames": ("frame_transition",),
    "manifolds": (
        "guess_manifold",
        "pt_embed",
        "pt_project",
        "pt_map",
        "scale_factors",
        "angle_between",
    ),
    "representations": (
        "cconvert",
        "guess_geometry_kind",
        "guess_rep",
        "guess_semantic_kind",
    ),
    "transforms": ("act", "compose", "simplify"),
}

FUNCTION_CASES = [
    (subpackage, name) for subpackage, names in API_FUNCTIONS.items() for name in names
]


@pytest.mark.parametrize("subpackage", API_FUNCTIONS)
def test_subpackage_importable(subpackage: str) -> None:
    """Each API subpackage is importable."""
    importlib.import_module(f"coordinaxs.api.{subpackage}")


@pytest.mark.parametrize(
    ("subpackage", "name"), FUNCTION_CASES, ids=[f"{s}.{n}" for s, n in FUNCTION_CASES]
)
def test_can_be_dispatched_on(subpackage: str, name: str) -> None:
    """Each API function has at least one registered dispatch method."""
    module = importlib.import_module(f"coordinaxs.api.{subpackage}")
    assert len(getattr(module, name).methods) > 0
