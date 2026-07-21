"""Unit tests for the optional-interop entry-point loader.

`tests/integration/frames/test_interop_import_order.py` covers the real
end-to-end behaviour, but it must spawn subprocesses (the session conftest
preloads `coordinax`, so import order cannot be varied in-process). These
in-process tests exercise the loader directly: it never raises (interop is an
optional extra), records failures for inspection, retries pending/failed entry
points, and honours the legacy group name.
"""

import warnings

from typing import Any

import pytest

import coordinax as cx


@pytest.fixture
def _reset_interop_state() -> Any:
    """Save/restore the loader's module-level state around a test."""
    state = cx._OPTIONAL_INTEROP_STATE
    saved = {
        "loading": state["loading"],
        "loaded": set(state["loaded"]),
        "failed": dict(state["failed"]),
    }
    yield
    state["loading"] = saved["loading"]
    state["loaded"] = saved["loaded"]
    state["failed"] = saved["failed"]


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


def _only_interop(ep: Any) -> Any:
    """An `entry_points` stub returning ``[ep]`` for the current interop group.

    Mirrors `importlib.metadata.entry_points()`: `group` is keyword-only and
    matched exactly, so a loader that queries the wrong group name gets nothing.
    """

    def entry_points(*, group: str) -> list[Any]:
        return [ep] if group == cx._INTEROP_ENTRYPOINT_GROUP else []

    return entry_points


# =============================================================================
# _load_optional_interop: success and bookkeeping


@pytest.mark.usefixtures("_reset_interop_state")
def test_loads_entry_point_once(monkeypatch: Any) -> None:
    """An entry point is loaded once and then recorded as loaded."""
    ep = _FakeEntryPoint("fake")
    monkeypatch.setattr(cx, "entry_points", _only_interop(ep))
    cx._OPTIONAL_INTEROP_STATE["loaded"] = set()
    cx._OPTIONAL_INTEROP_STATE["failed"] = {}

    cx._load_optional_interop()
    assert ep.loaded
    assert "fake" in cx._OPTIONAL_INTEROP_STATE["loaded"]
    assert cx._OPTIONAL_INTEROP_STATE["failed"] == {}

    ep.loaded = False
    cx._load_optional_interop()  # already recorded -> not re-loaded
    assert not ep.loaded


@pytest.mark.usefixtures("_reset_interop_state")
def test_reentrant_call_is_a_noop(monkeypatch: Any) -> None:
    """A re-entrant call returns immediately instead of double-loading."""
    ep = _FakeEntryPoint("fake")
    monkeypatch.setattr(cx, "entry_points", _only_interop(ep))
    cx._OPTIONAL_INTEROP_STATE["loaded"] = set()
    cx._OPTIONAL_INTEROP_STATE["failed"] = {}
    cx._OPTIONAL_INTEROP_STATE["loading"] = True

    cx._load_optional_interop()

    assert not ep.loaded


# =============================================================================
# _load_optional_interop: failures are recorded, never raised
#
# Interop is an optional extra, so a load failure must never break `import
# coordinax`. Instead of classifying failures (which previously required reading
# a private CPython import-state attribute), the loader records every failure
# and retries it on the next call. A transient failure (a sibling still
# mid-import in the astro->coordinax->interop->astro cycle) recovers on the
# retry; a genuine failure stays recorded. Behaviour is identical regardless of
# import order.


@pytest.mark.usefixtures("_reset_interop_state")
def test_failure_is_recorded_not_raised(monkeypatch: Any) -> None:
    """A load failure is recorded rather than raised, and is not marked loaded."""
    exc = RuntimeError("interop is broken")
    ep = _FakeEntryPoint("fake", exc=exc)
    monkeypatch.setattr(cx, "entry_points", _only_interop(ep))
    cx._OPTIONAL_INTEROP_STATE["loaded"] = set()
    cx._OPTIONAL_INTEROP_STATE["failed"] = {}

    cx._load_optional_interop()  # must not raise

    assert "fake" not in cx._OPTIONAL_INTEROP_STATE["loaded"]
    stored = cx._OPTIONAL_INTEROP_STATE["failed"]["fake"]
    assert stored is exc
    # The traceback is cleared so the long-lived `failed` dict does not pin the
    # stack frames (and their locals) captured when the load failed.
    assert stored.__traceback__ is None


@pytest.mark.usefixtures("_reset_interop_state")
def test_import_error_is_also_recorded_not_raised(monkeypatch: Any) -> None:
    """An ImportError (absent transitive dep) is recorded, not raised."""
    ep = _FakeEntryPoint("fake", exc=ImportError("no module named 'astropy'"))
    monkeypatch.setattr(cx, "entry_points", _only_interop(ep))
    cx._OPTIONAL_INTEROP_STATE["loaded"] = set()
    cx._OPTIONAL_INTEROP_STATE["failed"] = {}

    cx._load_optional_interop()  # must not raise

    assert "fake" not in cx._OPTIONAL_INTEROP_STATE["loaded"]
    assert isinstance(cx._OPTIONAL_INTEROP_STATE["failed"]["fake"], ImportError)


@pytest.mark.usefixtures("_reset_interop_state")
def test_retry_recovers_after_transient_failure(monkeypatch: Any) -> None:
    """An entry point that fails once then succeeds is recovered on the retry.

    This is the mechanism the astro-first import ordering relies on: the first
    load fails because a sibling is still mid-import, and a later call (from the
    sibling's own re-invocation hook) succeeds.
    """
    calls = {"n": 0}

    class _FlakyEntryPoint:
        name = "flaky"

        def __init__(self) -> None:
            self.loaded = False

        def load(self) -> object:
            calls["n"] += 1
            if calls["n"] == 1:  # first attempt fails, as if mid-cycle
                raise AttributeError("partially initialized: no 'Parallax' yet")
            self.loaded = True
            return object()

    ep = _FlakyEntryPoint()
    monkeypatch.setattr(cx, "entry_points", _only_interop(ep))
    cx._OPTIONAL_INTEROP_STATE["loaded"] = set()
    cx._OPTIONAL_INTEROP_STATE["failed"] = {}

    # Phase 1: fails -> recorded, not loaded.
    cx._load_optional_interop()
    assert not ep.loaded
    assert "flaky" not in cx._OPTIONAL_INTEROP_STATE["loaded"]
    assert "flaky" in cx._OPTIONAL_INTEROP_STATE["failed"]

    # Phase 2: a later call retries and now succeeds.
    cx._load_optional_interop()
    assert ep.loaded
    assert "flaky" in cx._OPTIONAL_INTEROP_STATE["loaded"]
    assert calls["n"] == 2


@pytest.mark.usefixtures("_reset_interop_state")
def test_success_clears_prior_failure(monkeypatch: Any) -> None:
    """Recovering an entry point removes it from the recorded failures."""
    calls = {"n": 0}

    class _FlakyEntryPoint:
        name = "flaky"

        def load(self) -> object:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("not yet")
            return object()

    ep = _FlakyEntryPoint()
    monkeypatch.setattr(cx, "entry_points", _only_interop(ep))
    cx._OPTIONAL_INTEROP_STATE["loaded"] = set()
    cx._OPTIONAL_INTEROP_STATE["failed"] = {}

    cx._load_optional_interop()
    assert "flaky" in cx._OPTIONAL_INTEROP_STATE["failed"]

    cx._load_optional_interop()
    assert "flaky" not in cx._OPTIONAL_INTEROP_STATE["failed"]
    assert "flaky" in cx._OPTIONAL_INTEROP_STATE["loaded"]


@pytest.mark.usefixtures("_reset_interop_state")
def test_one_failure_does_not_block_other_entry_points(monkeypatch: Any) -> None:
    """A failing entry point does not prevent a sibling from loading."""
    good = _FakeEntryPoint("good")
    bad = _FakeEntryPoint("bad", exc=RuntimeError("broken"))

    def entry_points(*, group: str) -> list[_FakeEntryPoint]:
        return [bad, good] if group == cx._INTEROP_ENTRYPOINT_GROUP else []

    monkeypatch.setattr(cx, "entry_points", entry_points)
    cx._OPTIONAL_INTEROP_STATE["loaded"] = set()
    cx._OPTIONAL_INTEROP_STATE["failed"] = {}

    cx._load_optional_interop()

    assert good.loaded
    assert "good" in cx._OPTIONAL_INTEROP_STATE["loaded"]
    assert "bad" in cx._OPTIONAL_INTEROP_STATE["failed"]


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
    cx._OPTIONAL_INTEROP_STATE["failed"] = {}

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
    cx._OPTIONAL_INTEROP_STATE["failed"] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cx._load_optional_interop()  # no warning: not legacy-only

    assert current_ep.loaded
    assert not legacy_ep.loaded
