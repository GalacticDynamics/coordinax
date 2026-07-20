"""Unit tests for the optional-interop entry-point loader.

`tests/integration/frames/test_interop_import_order.py` covers the real
end-to-end behaviour, but it must spawn subprocesses (the session conftest
preloads `coordinax`, so import order cannot be varied in-process). These
in-process tests exercise the loader's branches directly: readiness detection,
legacy-group handling, and the failure classification that decides whether an
entry point is left pending or raised.
"""

import sys
import warnings

import types
from typing import Any

import pytest

import coordinax as cx


@pytest.fixture
def _reset_interop_state() -> Any:
    """Save/restore the loader's module-level state around a test."""
    state = cx._OPTIONAL_INTEROP_STATE
    saved = {"loading": state["loading"], "loaded": set(state["loaded"])}
    yield
    state["loading"] = saved["loading"]
    state["loaded"] = saved["loaded"]


class _FakeEntryPoint:
    """Minimal stand-in for `importlib.metadata.EntryPoint`."""

    def __init__(self, name: str, exc: Exception | None = None) -> None:
        self.name = name
        self._exc = exc
        self.loaded = False

    def load(self) -> object:
        if self._exc is not None:
            raise self._exc
        self.loaded = True
        return object()


def _initializing_module(name: str) -> types.ModuleType:
    """A module object that reports itself as still executing its body."""
    module = types.ModuleType(name)
    spec = types.SimpleNamespace(_initializing=True)
    module.__spec__ = spec  # type: ignore[assignment]
    return module


# =============================================================================
# _coordinaxs_is_initializing


def test_not_initializing_at_rest() -> None:
    """With every coordinaxs module fully imported, nothing is initializing."""
    assert cx._coordinaxs_is_initializing() is False


def test_detects_partially_initialized_module(monkeypatch: Any) -> None:
    """A coordinaxs module mid-import is detected."""
    monkeypatch.setitem(
        sys.modules, "coordinaxs.fake_pkg", _initializing_module("coordinaxs.fake_pkg")
    )
    assert cx._coordinaxs_is_initializing() is True


def test_ignores_unrelated_initializing_module(monkeypatch: Any) -> None:
    """A non-coordinaxs module mid-import is not mistaken for the cycle."""
    monkeypatch.setitem(sys.modules, "unrelated", _initializing_module("unrelated"))
    assert cx._coordinaxs_is_initializing() is False


# =============================================================================
# _load_optional_interop


@pytest.mark.usefixtures("_reset_interop_state")
def test_loads_entry_point_once(monkeypatch: Any) -> None:
    """An entry point is loaded once and then recorded as loaded."""
    ep = _FakeEntryPoint("fake")
    monkeypatch.setattr(
        cx, "entry_points", lambda group: [ep] if "interop" in group else []
    )
    cx._OPTIONAL_INTEROP_STATE["loaded"] = set()

    cx._load_optional_interop()
    assert ep.loaded
    assert "fake" in cx._OPTIONAL_INTEROP_STATE["loaded"]

    ep.loaded = False
    cx._load_optional_interop()  # already recorded -> not re-loaded
    assert not ep.loaded


@pytest.mark.usefixtures("_reset_interop_state")
def test_failure_during_cycle_is_left_pending(monkeypatch: Any) -> None:
    """A failure while a coordinaxs module is mid-import leaves it pending."""
    ep = _FakeEntryPoint("fake", exc=AttributeError("no attribute 'Parallax'"))
    monkeypatch.setattr(
        cx, "entry_points", lambda group: [ep] if "interop" in group else []
    )
    monkeypatch.setitem(
        sys.modules, "coordinaxs.fake_pkg", _initializing_module("coordinaxs.fake_pkg")
    )
    cx._OPTIONAL_INTEROP_STATE["loaded"] = set()

    cx._load_optional_interop()  # must not raise

    assert "fake" not in cx._OPTIONAL_INTEROP_STATE["loaded"]


@pytest.mark.usefixtures("_reset_interop_state")
def test_genuine_failure_propagates(monkeypatch: Any) -> None:
    """A failure with nothing mid-import is real breakage and is raised."""
    ep = _FakeEntryPoint("fake", exc=RuntimeError("interop is broken"))
    monkeypatch.setattr(
        cx, "entry_points", lambda group: [ep] if "interop" in group else []
    )
    cx._OPTIONAL_INTEROP_STATE["loaded"] = set()

    with pytest.raises(RuntimeError, match="interop is broken"):
        cx._load_optional_interop()


@pytest.mark.usefixtures("_reset_interop_state")
def test_reentrant_call_is_a_noop(monkeypatch: Any) -> None:
    """A re-entrant call returns immediately instead of double-loading."""
    ep = _FakeEntryPoint("fake")
    monkeypatch.setattr(
        cx, "entry_points", lambda group: [ep] if "interop" in group else []
    )
    cx._OPTIONAL_INTEROP_STATE["loaded"] = set()
    cx._OPTIONAL_INTEROP_STATE["loading"] = True

    cx._load_optional_interop()

    assert not ep.loaded


# =============================================================================
# legacy entry-point group


@pytest.mark.usefixtures("_reset_interop_state")
def test_legacy_group_is_honoured_with_deprecation_warning(monkeypatch: Any) -> None:
    """An interop published under the pre-rename group still loads, and warns."""
    legacy_ep = _FakeEntryPoint("legacy")

    def fake_entry_points(*, group: str) -> list[_FakeEntryPoint]:
        return [legacy_ep] if group == cx._LEGACY_INTEROP_ENTRYPOINT_GROUP else []

    monkeypatch.setattr(cx, "entry_points", fake_entry_points)
    cx._OPTIONAL_INTEROP_STATE["loaded"] = set()

    with pytest.warns(DeprecationWarning, match="legacy"):
        cx._load_optional_interop()

    assert legacy_ep.loaded


@pytest.mark.usefixtures("_reset_interop_state")
def test_current_group_wins_over_legacy_duplicate(monkeypatch: Any) -> None:
    """A distribution in both groups is loaded once, from the current group."""
    current_ep = _FakeEntryPoint("dup")
    legacy_ep = _FakeEntryPoint("dup")

    def fake_entry_points(*, group: str) -> list[_FakeEntryPoint]:
        if group == cx._INTEROP_ENTRYPOINT_GROUP:
            return [current_ep]
        return [legacy_ep]

    monkeypatch.setattr(cx, "entry_points", fake_entry_points)
    cx._OPTIONAL_INTEROP_STATE["loaded"] = set()

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cx._load_optional_interop()  # no warning: not legacy-only

    assert current_ep.loaded
    assert not legacy_ep.loaded
