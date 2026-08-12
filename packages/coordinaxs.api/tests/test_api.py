"""Tests for the `coordinaxs.api` dispatch surface.

`coordinaxs.api` exists to declare the multiple-dispatch functions that the
implementation packages register against. There is nothing to exercise beyond
"the subpackage imports" and "the function has at least one method", so the
whole surface is one table.
"""

__all__: tuple[str, ...] = ()

import importlib

import pytest

#: The `coordinaxs.api` subpackages. The dispatch functions themselves are read
#: off each subpackage's ``__all__`` rather than listed here: a hand-maintained
#: copy silently stops covering whatever is added later.
SUBPACKAGES = ("charts", "frames", "manifolds", "representations", "transforms")


def _api_functions() -> list[tuple[str, str]]:
    """Every ``(subpackage, name)`` pair exported by `coordinaxs.api`."""
    return [
        (subpackage, name)
        for subpackage in SUBPACKAGES
        for name in importlib.import_module(f"coordinaxs.api.{subpackage}").__all__
    ]


FUNCTION_CASES = _api_functions()


@pytest.mark.parametrize("subpackage", SUBPACKAGES)
def test_subpackage_importable(subpackage: str) -> None:
    """Each API subpackage is importable."""
    importlib.import_module(f"coordinaxs.api.{subpackage}")


@pytest.mark.parametrize(
    ("subpackage", "name"), FUNCTION_CASES, ids=[f"{s}.{n}" for s, n in FUNCTION_CASES]
)
def test_can_be_dispatched_on(subpackage: str, name: str) -> None:
    """Each API function has at least one registered dispatch method.

    An abstract with no registered method is not an extension point, it is a
    guaranteed `NotImplementedError` at the call site.
    """
    module = importlib.import_module(f"coordinaxs.api.{subpackage}")
    assert len(getattr(module, name).methods) > 0
