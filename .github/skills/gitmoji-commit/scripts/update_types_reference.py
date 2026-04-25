"""Sync gitmoji type reference with cz-conventional-gitmoji metadata."""

# ruff: noqa: INP001

import argparse
import re
import sys
import tomllib
from importlib.metadata import version as pkg_version
from pathlib import Path

from shared.spec import mojis

REPO_ROOT = Path(__file__).resolve().parents[4]
PYPROJECT = REPO_ROOT / "pyproject.toml"
UV_LOCK = REPO_ROOT / "uv.lock"
REFERENCE_FILE = REPO_ROOT / ".github/skills/gitmoji-commit/references/types.md"


def get_required_specifier_from_pyproject() -> str:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dev_group = data["dependency-groups"]["dev"]
    for item in dev_group:
        if isinstance(item, str) and item.startswith("cz-conventional-gitmoji"):
            return item.removeprefix("cz-conventional-gitmoji") or "(unbounded)"
    raise RuntimeError(
        "Could not find cz-conventional-gitmoji entry in dependency-groups.dev"
    )


def get_locked_version_from_uv_lock() -> str:
    text = UV_LOCK.read_text(encoding="utf-8")
    match = re.search(
        r"\[\[package\]\]\nname = \"cz-conventional-gitmoji\"\nversion = \"([^\"]+)\"",
        text,
    )
    if not match:
        raise RuntimeError("Could not find cz-conventional-gitmoji in uv.lock")
    return match.group(1)


def get_current_reference_versions(text: str) -> tuple[str | None, str | None]:
    required_match = re.search(r"- Required by pyproject.toml: `([^`]+)`", text)
    locked_match = re.search(r"- Locked in uv.lock: `([^`]+)`", text)
    required = required_match.group(1) if required_match else None
    locked = locked_match.group(1) if locked_match else None
    return required, locked


def render_reference(
    required_specifier: str, locked_version: str, installed_version: str
) -> str:
    source_pkg = (
        "- Source package installed for generation: "
        f"`cz-conventional-gitmoji=={installed_version}`"
    )
    lines = [
        "# cz_gitmoji Type Reference",
        "",
        "Generated from `shared/spec.py` in the `cz_gitmoji` package.",
        "",
        f"- Required by pyproject.toml: `{required_specifier}`",
        f"- Locked in uv.lock: `{locked_version}`",
        source_pkg,
        "",
        "Commit prefix format: `{icon} {type}` (with a space between icon and type).",
        "",
        "| Icon | Type | Description |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        (f"| {moji['icon']} | `{moji['type']}` | {normalize_desc(moji['desc'])} |")
        for moji in mojis
    )
    lines.append("")
    return "\n".join(lines)


def normalize_desc(description: str) -> str:
    """Normalize known upstream typos to satisfy local spell checks."""
    misspelled = "".join(("misce", "laneous"))  # noqa: FLY002
    return description.replace(misspelled, "miscellaneous")


def out(message: str) -> None:
    """Write a single line to stdout without using print."""
    sys.stdout.write(f"{message}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--check", action="store_true", help="Return non-zero if reference is stale"
    )
    modes.add_argument("--write", action="store_true", help="Rewrite reference file")
    args = parser.parse_args()

    required_specifier = get_required_specifier_from_pyproject()
    locked_version = get_locked_version_from_uv_lock()
    installed_version = pkg_version("cz-conventional-gitmoji")

    expected = render_reference(required_specifier, locked_version, installed_version)
    current = (
        REFERENCE_FILE.read_text(encoding="utf-8") if REFERENCE_FILE.exists() else ""
    )

    current_required, current_locked = get_current_reference_versions(current)
    stale_reason = []
    if current_required != required_specifier:
        stale_reason.append(
            "required specifier changed "
            f"(current={current_required!r}, expected={required_specifier!r})"
        )
    if current_locked != locked_version:
        stale_reason.append(
            "locked version changed "
            f"(current={current_locked!r}, expected={locked_version!r})"
        )
    if current != expected:
        stale_reason.append("reference content differs from generated output")

    is_stale = bool(stale_reason)

    if args.check:
        if is_stale:
            out("Reference is stale:")
            for reason in stale_reason:
                out(f"- {reason}")
            return 1
        out("Reference is up to date.")
        return 0

    if args.write:
        REFERENCE_FILE.write_text(expected, encoding="utf-8")
        if is_stale:
            out("Updated reference file:")
            for reason in stale_reason:
                out(f"- {reason}")
        else:
            out("Reference was already up to date; file normalized.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
