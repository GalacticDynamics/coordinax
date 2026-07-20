"""Legacy entry-point group back-compat for frames and transforms.

The `coordinaxs.{frames,transforms}` group names are cross-distribution
contracts. Renaming them from `coordinax.*` would silently drop third-party
registrants, so both consumers still read the legacy name and emit a
`DeprecationWarning`. Nothing in-tree registers under the legacy names, so
these tests supply fake entry points.
"""

import warnings

from typing import Any

import pytest

import coordinax.frames as cxf
import coordinax.transforms as cxfm


class _FakeEntryPoint:
    """Minimal stand-in for `importlib.metadata.EntryPoint`."""

    def __init__(self, name: str, exports: dict[str, object] | None = None) -> None:
        self.name = name
        self._exports = {} if exports is None else exports
        self.loaded = False

    def load(self) -> Any:
        self.loaded = True
        return lambda: self._exports


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
