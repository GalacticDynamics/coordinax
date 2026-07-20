"""Unit tests for the optional-interop entry-point loader.

`tests/integration/frames/test_interop_import_order.py` covers the real
end-to-end behaviour, but it must spawn subprocesses (the session conftest
preloads `coordinax`, so import order cannot be varied in-process). These
in-process tests exercise the loader's branches directly: readiness detection,
legacy-group handling, and the failure classification that decides whether an
entry point is left pending or raised.
"""

import importlib
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


def test_importlib_marks_initializing_modules(tmp_path: Any) -> None:
    """Pin the private importlib contract `_coordinaxs_is_initializing` uses.

    `_coordinaxs_is_initializing` reads ``__spec__._initializing`` — a private,
    undocumented CPython importlib attribute that is ``True`` only while a
    module's body executes. Every other test in this module *fakes* that
    attribute, so a Python release that removed or renamed it would leave the
    faked tests green while the real cycle-tolerance silently broke (astro-first
    import would start raising). This test imports a real module that captures
    its own ``__spec__._initializing`` during execution, pinning the contract so
    such a change is caught here instead.
    """
    probe = tmp_path / "_probe_initializing.py"
    probe.write_text(
        "import sys\n"
        "_spec = sys.modules[__name__].__spec__\n"
        "captured = getattr(_spec, '_initializing', 'MISSING')\n"
    )
    probe_dir = str(tmp_path)
    sys.path.insert(0, probe_dir)
    try:
        module = importlib.import_module("_probe_initializing")
        assert module.captured is True, (
            "importlib no longer sets __spec__._initializing during module "
            "execution; _coordinaxs_is_initializing() needs updating."
        )
    finally:
        sys.path.remove(probe_dir)
        sys.modules.pop("_probe_initializing", None)


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
def test_retry_completes_after_cycle_clears(monkeypatch: Any) -> None:
    """A pending entry point loads on a later call once the cycle clears.

    This is the whole point of the retryable design: fail while a sibling is
    mid-import, then succeed on a later at-rest call. The astro-first ordering
    relies on exactly this two-phase transition.
    """
    calls = {"n": 0}

    class _FlakyEntryPoint:
        name = "flaky"

        def __init__(self) -> None:
            self.loaded = False

        def load(self) -> object:
            calls["n"] += 1
            if calls["n"] == 1:  # first attempt: fail as if mid-cycle
                raise AttributeError("partially initialized: no 'Parallax' yet")
            self.loaded = True
            return object()

    ep = _FlakyEntryPoint()
    monkeypatch.setattr(
        cx, "entry_points", lambda group: [ep] if "interop" in group else []
    )
    cx._OPTIONAL_INTEROP_STATE["loaded"] = set()

    # Phase 1: a coordinaxs module is initializing -> the failure is tolerated
    # and the entry point is left pending.
    initializing = _initializing_module("coordinaxs.fake_pkg")
    monkeypatch.setitem(sys.modules, "coordinaxs.fake_pkg", initializing)
    cx._load_optional_interop()
    assert not ep.loaded
    assert "flaky" not in cx._OPTIONAL_INTEROP_STATE["loaded"]

    # Phase 2: the sibling finished initializing; a later call retries and the
    # entry point now loads and is recorded.
    initializing.__spec__._initializing = False  # type: ignore[union-attr]
    cx._load_optional_interop()
    assert ep.loaded
    assert "flaky" in cx._OPTIONAL_INTEROP_STATE["loaded"]
    assert calls["n"] == 2


@pytest.mark.usefixtures("_reset_interop_state")
def test_genuine_error_is_swallowed_while_initializing(monkeypatch: Any) -> None:
    """A genuinely broken interop is swallowed, not raised, while initializing.

    This pins the loader's documented limitation.
    `_load_optional_interop` cannot distinguish a real failure from the expected
    import cycle while a `coordinaxs` module is still initializing, so it leaves
    the entry point pending instead of raising (see the loader docstring). This
    test locks that intended-but-imperfect semantics so a future change to the
    failure classification is caught.
    """
    ep = _FakeEntryPoint("fake", exc=RuntimeError("interop is genuinely broken"))
    monkeypatch.setattr(
        cx, "entry_points", lambda group: [ep] if "interop" in group else []
    )
    monkeypatch.setitem(
        sys.modules, "coordinaxs.fake_pkg", _initializing_module("coordinaxs.fake_pkg")
    )
    cx._OPTIONAL_INTEROP_STATE["loaded"] = set()

    # Swallowed, not raised, because a coordinaxs module is still initializing.
    cx._load_optional_interop()

    assert "fake" not in cx._OPTIONAL_INTEROP_STATE["loaded"]


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
