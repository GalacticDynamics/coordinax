"""Shared loader for optional entry-point export groups (frames, transforms)."""

__all__: tuple[str, ...] = ()

import warnings
from importlib.metadata import entry_points

from collections.abc import Mapping
from typing import Any


def resolve_entrypoints_with_legacy(
    group: str, legacy_group: str, /, *, warn_what: str
) -> list[Any]:
    """Resolve entry points from ``group``, honouring a pre-rename ``legacy_group``.

    The group is a cross-distribution contract, so third-party registrants
    published against ``legacy_group`` are still honoured -- with a
    `DeprecationWarning` naming ``warn_what`` (e.g. ``"frame exports"``) -- rather
    than silently dropped. An entry point present under both groups is taken
    from ``group`` only (no duplicate load). Returns the combined entry points,
    sorted by name.
    """
    current = list(entry_points(group=group))
    seen = {ep.name for ep in current}
    legacy = [ep for ep in entry_points(group=legacy_group) if ep.name not in seen]
    if legacy:
        names = ", ".join(sorted(ep.name for ep in legacy))
        warnings.warn(
            f"Entry point(s) {names} register {warn_what} under the legacy "
            f"'{legacy_group}' group. That group is deprecated; publish under "
            f"'{group}' instead. Support for the legacy group will be removed "
            "in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )
    return sorted(current + legacy, key=lambda ep: ep.name)


def load_exports(
    entrypoints: list[Any], /, *, group: str, noun: str
) -> dict[str, object]:
    """Load and validate optional exports from entry points.

    Each entry point must load a callable that returns a string-keyed mapping;
    conflicting exports (same name, different value) are rejected. ``group``
    names the entry-point group in the validation messages and ``noun`` (e.g.
    ``"frame export"``) is used in the conflict message. Returns the merged
    export mapping; the caller injects it into its own namespace.
    """
    exported: dict[str, object] = {}
    export_owners: dict[str, str] = {}
    for ep in entrypoints:
        provider = ep.load()
        if not callable(provider):
            msg = f"Entry point {ep.name!r} in group '{group}' is not callable."
            raise TypeError(msg)
        exports = provider()
        if not isinstance(exports, Mapping):
            msg = f"Entry point {ep.name!r} in group '{group}' must return a mapping."
            raise TypeError(msg)
        for name, value in exports.items():
            if not isinstance(name, str):
                msg = (
                    f"Entry point {ep.name!r} in group '{group}' produced a "
                    "non-string export name."
                )
                raise TypeError(msg)
            if name in exported and exported[name] is not value:
                msg = (
                    f"Conflicting {noun} {name!r} from entry points "
                    f"{export_owners[name]!r} and {ep.name!r}."
                )
                raise RuntimeError(msg)
            exported[name] = value
            export_owners[name] = ep.name
    return exported
