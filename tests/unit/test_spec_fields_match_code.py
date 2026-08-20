"""The spec's **Fields** tables must name the dataclass fields that exist.

`docs/spec.md` is the authoritative description of the public API, but nothing
tied its **Fields** entries to the code, so they drifted: `Scale` was documented
with `factor`/`chart` long after it stored a matrix (#750), `Shear` likewise
(#762), and `Tangent` listed a `manifold` field that never existed.

Only the per-type sections are checked. The exported-objects *table* earlier in
the spec is a curated summary -- the `coordinax.charts` row deliberately omits
50-odd class names -- so it is not an `__all__` mirror and is left alone.
"""

__all__: tuple[str, ...] = ()

import dataclasses
import importlib
import re
from pathlib import Path

import pytest

SPEC = Path(__file__).parents[2] / "docs" / "spec.md"

# Where a documented type name might live. Searched in order; the first hit that
# is a dataclass wins.
_NAMESPACES = (
    "coordinax",
    "coordinax.vectors",
    "coordinax.transforms",
    "coordinax.charts",
    "coordinax.manifolds",
    "coordinax.representations",
    "coordinax.frames",
    "coordinax.angles",
    "coordinax.distances",
)

# A `**Fields:**` block, up to the next bold heading or the end of the section.
_FIELDS_BLOCK = re.compile(
    r"^\s+\*\*Fields:?\*\*:?\s*\n(.*?)(?=^\s+\*\*|\Z)", re.MULTILINE | re.DOTALL
)
# Two spellings are in use: a bullet list, and a markdown table.
_BULLET = re.compile(r"^\s+-\s+`([A-Za-z_]\w*)\s*[:`]", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\s+\|\s*`([A-Za-z_]\w*)`\s*\|", re.MULTILINE)


def _documented() -> list[tuple[str, list[str]]]:
    """Every ``!!! info `Name`` section that makes a **Fields** claim."""
    text = SPEC.read_text(encoding="utf-8")
    out = []
    for block in re.split(r"^!!! info `", text, flags=re.MULTILINE)[1:]:
        name = block.split("`")[0]
        found = _FIELDS_BLOCK.search(block)
        if found is None:
            continue
        body = found.group(1)
        fields = _BULLET.findall(body) + _TABLE_ROW.findall(body)
        if fields:
            out.append((name, fields))
    return out


def _resolve(name: str) -> type | None:
    for ns in _NAMESPACES:
        obj = getattr(importlib.import_module(ns), name, None)
        if isinstance(obj, type) and dataclasses.is_dataclass(obj):
            return obj
    return None


DOCUMENTED = _documented()


def test_the_spec_still_has_fields_sections_to_check() -> None:
    """Guard the parser: a regex that silently matches nothing proves nothing."""
    assert len(DOCUMENTED) >= 8, DOCUMENTED


@pytest.mark.parametrize(("name", "fields"), DOCUMENTED, ids=[n for n, _ in DOCUMENTED])
def test_documented_fields_exist_on_the_dataclass(name: str, fields: list[str]) -> None:
    """Compared as *sets*, not sequences.

    Order is deliberately not enforced. `Translate` defines its own `__init__`,
    whose parameter order is the useful one to document and differs from
    `dataclasses.fields`; requiring the latter would fail a section that is
    right.
    """
    cls = _resolve(name)
    if cls is None:
        pytest.skip(f"{name} is not a resolvable dataclass")

    actual = {f.name for f in dataclasses.fields(cls)}
    documented = set(fields)

    assert not (documented - actual), (
        f"{name}: spec documents field(s) the class does not have: "
        f"{sorted(documented - actual)}"
    )
    assert not (actual - documented), (
        f"{name}: class has field(s) the spec does not document: "
        f"{sorted(actual - documented)}"
    )
