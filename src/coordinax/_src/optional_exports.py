"""Shared loader for optional entry-point export groups (frames, transforms)."""

__all__: tuple[str, ...] = ()

from collections.abc import Mapping
from typing import Any


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
