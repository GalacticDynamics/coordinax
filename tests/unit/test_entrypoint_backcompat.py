"""Legacy entry-point group back-compat for frames and transforms.

The `coordinaxs.{frames,transforms}` group names are cross-distribution
contracts. Renaming them from `coordinax.*` would silently drop third-party
registrants, so both consumers still read the legacy name and emit a
`DeprecationWarning`. Nothing in-tree registers under the legacy names, so
these tests supply fake entry points.
"""

import warnings

from collections.abc import Callable, Iterator
from typing import Any

import pytest

import coordinax.frames as cxf
import coordinax.transforms as cxfm

_UNSET = object()


class _FakeEntryPoint:
    """Minimal stand-in for `importlib.metadata.EntryPoint`.

    By default ``load()`` returns a provider callable that yields ``exports``.
    Pass ``provider`` to override what ``load()`` returns (e.g. a non-callable,
    or a callable that returns a non-mapping) to drive the loader's validation
    branches.
    """

    def __init__(
        self,
        name: str,
        exports: dict[object, object] | None = None,
        provider: Any = _UNSET,
    ) -> None:
        self.name = name
        self._exports = {} if exports is None else exports
        self._provider = provider
        self.loaded = False

    def load(self) -> Any:
        self.loaded = True
        if self._provider is not _UNSET:
            return self._provider
        return lambda: self._exports


def _fake_groups(**by_group: list[_FakeEntryPoint]) -> Callable[..., list]:
    """Build an `entry_points`-compatible stub keyed by group name."""

    def entry_points(*, group: str) -> list[_FakeEntryPoint]:
        return by_group.get(group, [])

    return entry_points


@pytest.fixture
def _isolate_frames() -> Iterator[None]:
    """Reset the frames loader state and drop any globals a test injects."""
    before = set(vars(cxf))
    cxf._OPTIONAL_FRAME_EXPORTS_STATE["loading"] = False
    yield
    cxf._OPTIONAL_FRAME_EXPORTS_STATE["loading"] = False
    for name in set(vars(cxf)) - before:
        delattr(cxf, name)


@pytest.fixture
def _isolate_transforms() -> Iterator[None]:
    """Reset the transforms loader state and drop any globals a test injects."""
    before = set(vars(cxfm))
    cxfm._OPTIONAL_TRANSFORM_EXPORTS_STATE["loading"] = False
    yield
    cxfm._OPTIONAL_TRANSFORM_EXPORTS_STATE["loading"] = False
    for name in set(vars(cxfm)) - before:
        delattr(cxfm, name)


# =============================================================================
# frames


def test_frames_legacy_group_warns_and_is_included(monkeypatch: Any) -> None:
    """A registrant under the pre-rename frames group is still returned."""
    legacy = _FakeEntryPoint("legacyframes")

    def fake_entry_points(*, group: str) -> list[_FakeEntryPoint]:
        return [legacy] if group == cxf._LEGACY_FRAME_EXPORTS_ENTRYPOINT_GROUP else []

    monkeypatch.setattr(cxf, "entry_points", fake_entry_points)

    with pytest.warns(DeprecationWarning, match="legacy"):
        eps = cxf._frame_export_entrypoints()

    assert [ep.name for ep in eps] == ["legacyframes"]


def test_frames_current_group_does_not_warn(monkeypatch: Any) -> None:
    """A registrant under the current group produces no deprecation warning."""
    current = _FakeEntryPoint("astro")

    def fake_entry_points(*, group: str) -> list[_FakeEntryPoint]:
        return [current] if group == cxf._FRAME_EXPORTS_ENTRYPOINT_GROUP else []

    monkeypatch.setattr(cxf, "entry_points", fake_entry_points)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        eps = cxf._frame_export_entrypoints()

    assert [ep.name for ep in eps] == ["astro"]


def test_frames_duplicate_prefers_current_group(monkeypatch: Any) -> None:
    """A name in both groups is taken once, from the current group."""
    current = _FakeEntryPoint("dup")
    legacy = _FakeEntryPoint("dup")

    def fake_entry_points(*, group: str) -> list[_FakeEntryPoint]:
        if group == cxf._FRAME_EXPORTS_ENTRYPOINT_GROUP:
            return [current]
        return [legacy]

    monkeypatch.setattr(cxf, "entry_points", fake_entry_points)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        eps = cxf._frame_export_entrypoints()

    assert eps == [current]


# =============================================================================
# transforms


def test_transforms_legacy_group_warns_and_loads(monkeypatch: Any) -> None:
    """A registrant under the pre-rename transforms group is still loaded."""
    legacy = _FakeEntryPoint("legacyxfm", exports={})

    def fake_entry_points(*, group: str) -> list[_FakeEntryPoint]:
        if group == cxfm._LEGACY_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP:
            return [legacy]
        return []

    monkeypatch.setattr(cxfm, "entry_points", fake_entry_points)

    with pytest.warns(DeprecationWarning, match="legacy"):
        cxfm._load_optional_transform_exports()

    assert legacy.loaded


def test_transforms_current_group_does_not_warn(monkeypatch: Any) -> None:
    """A registrant under the current transforms group produces no warning."""
    current = _FakeEntryPoint("xfm", exports={})

    def fake_entry_points(*, group: str) -> list[_FakeEntryPoint]:
        return [current] if group == cxfm._TRANSFORM_EXPORTS_ENTRYPOINT_GROUP else []

    monkeypatch.setattr(cxfm, "entry_points", fake_entry_points)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cxfm._load_optional_transform_exports()

    assert current.loaded


# =============================================================================
# frames loader: validation, conflict, and globals landing


@pytest.mark.usefixtures("_isolate_frames")
def test_frames_provider_not_callable_raises(monkeypatch: Any) -> None:
    """An entry point that loads to a non-callable is rejected."""
    ep = _FakeEntryPoint("bad", provider=42)  # load() returns a non-callable
    monkeypatch.setattr(
        cxf,
        "entry_points",
        _fake_groups(**{cxf._FRAME_EXPORTS_ENTRYPOINT_GROUP: [ep]}),
    )
    with pytest.raises(TypeError, match="is not callable"):
        cxf._load_optional_frame_exports()


@pytest.mark.usefixtures("_isolate_frames")
def test_frames_exports_not_mapping_raises(monkeypatch: Any) -> None:
    """A provider that returns a non-mapping is rejected."""
    ep = _FakeEntryPoint("bad", provider=lambda: [1, 2, 3])
    monkeypatch.setattr(
        cxf,
        "entry_points",
        _fake_groups(**{cxf._FRAME_EXPORTS_ENTRYPOINT_GROUP: [ep]}),
    )
    with pytest.raises(TypeError, match="must return a mapping"):
        cxf._load_optional_frame_exports()


@pytest.mark.usefixtures("_isolate_frames")
def test_frames_non_string_export_name_raises(monkeypatch: Any) -> None:
    """A non-string export name is rejected."""
    ep = _FakeEntryPoint("bad", exports={123: object()})
    monkeypatch.setattr(
        cxf,
        "entry_points",
        _fake_groups(**{cxf._FRAME_EXPORTS_ENTRYPOINT_GROUP: [ep]}),
    )
    with pytest.raises(TypeError, match="non-string export name"):
        cxf._load_optional_frame_exports()


@pytest.mark.usefixtures("_isolate_frames")
def test_frames_conflicting_exports_raise(monkeypatch: Any) -> None:
    """Two entry points exporting the same name with different values conflict."""
    ep_a = _FakeEntryPoint("aaa", exports={"Dup": object()})
    ep_b = _FakeEntryPoint("bbb", exports={"Dup": object()})
    monkeypatch.setattr(
        cxf,
        "entry_points",
        _fake_groups(**{cxf._FRAME_EXPORTS_ENTRYPOINT_GROUP: [ep_a, ep_b]}),
    )
    with pytest.raises(RuntimeError, match="Conflicting frame export 'Dup'"):
        cxf._load_optional_frame_exports()


@pytest.mark.usefixtures("_isolate_frames")
def test_frames_same_value_from_two_groups_is_not_a_conflict(monkeypatch: Any) -> None:
    """The same object under one name in both groups is not a conflict."""
    shared = object()
    current = _FakeEntryPoint("dup", exports={"Shared": shared})
    legacy = _FakeEntryPoint("dup", exports={"Shared": shared})
    monkeypatch.setattr(
        cxf,
        "entry_points",
        _fake_groups(
            **{
                cxf._FRAME_EXPORTS_ENTRYPOINT_GROUP: [current],
                cxf._LEGACY_FRAME_EXPORTS_ENTRYPOINT_GROUP: [legacy],
            }
        ),
    )
    # `dup` is present in both groups, so the current one wins the dedup and the
    # legacy one is never even loaded -> no conflict, no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cxf._load_optional_frame_exports()
    assert cxf.Shared is shared


@pytest.mark.usefixtures("_isolate_frames")
def test_frames_exports_land_in_module_globals(monkeypatch: Any) -> None:
    """A valid export is injected into the `coordinax.frames` namespace."""
    sentinel = object()
    ep = _FakeEntryPoint("plugin", exports={"_InjectedFrame": sentinel})
    monkeypatch.setattr(
        cxf,
        "entry_points",
        _fake_groups(**{cxf._FRAME_EXPORTS_ENTRYPOINT_GROUP: [ep]}),
    )
    cxf._load_optional_frame_exports()
    assert cxf._InjectedFrame is sentinel  # cleaned up by the fixture


# =============================================================================
# transforms loader: validation, conflict, dedup, and globals landing


@pytest.mark.usefixtures("_isolate_transforms")
def test_transforms_provider_not_callable_raises(monkeypatch: Any) -> None:
    """An entry point that loads to a non-callable is rejected."""
    ep = _FakeEntryPoint("bad", provider=42)
    monkeypatch.setattr(
        cxfm,
        "entry_points",
        _fake_groups(**{cxfm._TRANSFORM_EXPORTS_ENTRYPOINT_GROUP: [ep]}),
    )
    with pytest.raises(TypeError, match="is not callable"):
        cxfm._load_optional_transform_exports()


@pytest.mark.usefixtures("_isolate_transforms")
def test_transforms_exports_not_mapping_raises(monkeypatch: Any) -> None:
    """A provider that returns a non-mapping is rejected."""
    ep = _FakeEntryPoint("bad", provider=lambda: [1, 2, 3])
    monkeypatch.setattr(
        cxfm,
        "entry_points",
        _fake_groups(**{cxfm._TRANSFORM_EXPORTS_ENTRYPOINT_GROUP: [ep]}),
    )
    with pytest.raises(TypeError, match="must return a mapping"):
        cxfm._load_optional_transform_exports()


@pytest.mark.usefixtures("_isolate_transforms")
def test_transforms_non_string_export_name_raises(monkeypatch: Any) -> None:
    """A non-string export name is rejected."""
    ep = _FakeEntryPoint("bad", exports={123: object()})
    monkeypatch.setattr(
        cxfm,
        "entry_points",
        _fake_groups(**{cxfm._TRANSFORM_EXPORTS_ENTRYPOINT_GROUP: [ep]}),
    )
    with pytest.raises(TypeError, match="non-string export name"):
        cxfm._load_optional_transform_exports()


@pytest.mark.usefixtures("_isolate_transforms")
def test_transforms_conflicting_exports_raise(monkeypatch: Any) -> None:
    """Two entry points exporting the same name with different values conflict."""
    ep_a = _FakeEntryPoint("aaa", exports={"Dup": object()})
    ep_b = _FakeEntryPoint("bbb", exports={"Dup": object()})
    monkeypatch.setattr(
        cxfm,
        "entry_points",
        _fake_groups(**{cxfm._TRANSFORM_EXPORTS_ENTRYPOINT_GROUP: [ep_a, ep_b]}),
    )
    with pytest.raises(RuntimeError, match="Conflicting transform export 'Dup'"):
        cxfm._load_optional_transform_exports()


@pytest.mark.usefixtures("_isolate_transforms")
def test_transforms_duplicate_prefers_current_group(monkeypatch: Any) -> None:
    """A name in both transform groups is loaded once, from the current group."""
    current = _FakeEntryPoint("dup", exports={})
    legacy = _FakeEntryPoint("dup", exports={})
    monkeypatch.setattr(
        cxfm,
        "entry_points",
        _fake_groups(
            **{
                cxfm._TRANSFORM_EXPORTS_ENTRYPOINT_GROUP: [current],
                cxfm._LEGACY_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP: [legacy],
            }
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cxfm._load_optional_transform_exports()
    assert current.loaded
    assert not legacy.loaded


@pytest.mark.usefixtures("_isolate_transforms")
def test_transforms_exports_land_in_module_globals(monkeypatch: Any) -> None:
    """A valid export is injected into the `coordinax.transforms` namespace."""
    sentinel = object()
    ep = _FakeEntryPoint("plugin", exports={"_InjectedTransform": sentinel})
    monkeypatch.setattr(
        cxfm,
        "entry_points",
        _fake_groups(**{cxfm._TRANSFORM_EXPORTS_ENTRYPOINT_GROUP: [ep]}),
    )
    cxfm._load_optional_transform_exports()
    assert cxfm._InjectedTransform is sentinel  # cleaned up by the fixture


@pytest.mark.usefixtures("_isolate_transforms")
def test_transforms_reentrant_call_is_a_noop(monkeypatch: Any) -> None:
    """A re-entrant transforms load returns immediately without loading."""
    ep = _FakeEntryPoint("xfm", exports={})
    monkeypatch.setattr(
        cxfm,
        "entry_points",
        _fake_groups(**{cxfm._TRANSFORM_EXPORTS_ENTRYPOINT_GROUP: [ep]}),
    )
    cxfm._OPTIONAL_TRANSFORM_EXPORTS_STATE["loading"] = True
    cxfm._load_optional_transform_exports()
    assert not ep.loaded
