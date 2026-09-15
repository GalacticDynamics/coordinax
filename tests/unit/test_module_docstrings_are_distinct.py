"""A copied module docstring makes a module describe another file's job.

Twenty-nine modules across five packages opened with a line belonging to some
other file -- `transforms/_src/groups.py` saying "Frames sub-package",
`charts/_src/charts_product.py` saying "Hypothesis strategies for coordinax
representations", three separate `constants.py` saying "Internal custom types".
Each arrived the same way: a file was copied and its docstring was not
rewritten. See #875, #877 and #903.

The signature is one first line appearing in two different directories. That
is not itself a fault -- parallel registration modules legitimately share one,
and `SHARED_BY_DESIGN` lists the cases that do -- so this pins the set instead
of forbidding it. A *new* shared line fails, and the fix is either to write a
docstring that describes the file or to add the line here with a reason.

It reads the first line only, which is a real gap: #903 corrected
`transforms/_src/groups.py`'s summary and left its body still claiming to be
"the private implementation of the frames sub-package", and this would not
have noticed. Widening it to the body wants a rule for prose that merely
mentions another sub-package ("reference frames"), which the first line
rarely does.
"""

__all__: tuple[str, ...] = ()

import ast
import collections
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SEARCH = (_ROOT / "src", _ROOT / "packages")

#: First lines shared across directories on purpose, and why.
SHARED_BY_DESIGN: dict[str, str] = {
    "Register ``metric_matrix`` and ``metric_representation`` dispatch rules.": (
        "one per manifold family; each registers the same two dispatches"
    ),
    "Point-roled transformations in the same atlas.": (
        "`register_ptmap.py` in charts and in spherical, same job per atlas"
    ),
    "Optional dependencies. Internal use only.": "the per-package optional-deps shim",
    "Register `plum.convert` to/from distances.": (
        "distance converters, registered from core and from astro"
    ),
    "``import coordinaxs.astro as cxastro`` — Frames for Astronomy.": (
        "the astro package docstring, repeated on its `_src` re-export"
    ),
    "Utilities.": "the annotation helpers under `hypothesis/utils/_src`",
    "Internal custom types.": "per-package `custom_types.py`",
    "Manifolds in coordinax.": "the manifold sub-package `__init__` files",
    "Hypothesis strategies for coordinax.": "the strategy sub-package `__init__` files",
    "Hypothesis strategies for Distance quantities.": "the distance strategy modules",
    "Hypothesis strategies for CDict objects.": (
        "chart and representation CDict strategies"
    ),
    "Hypothesis strategies for coordinax manifolds.": (
        "manifold strategies and their `_src`"
    ),
    "Hypothesis strategies for coordinax vectors.": (
        "vector strategies and their `_src`"
    ),
}


def _first_lines() -> dict[str, list[str]]:
    """Every module's docstring first line, mapped to the files that use it."""
    out: dict[str, list[str]] = collections.defaultdict(list)
    for root in _SEARCH:
        for path in sorted(root.rglob("*.py")):
            posix = path.as_posix()
            if "__pycache__" in posix or path.name == "_version.py":
                continue
            if "/tests/" in posix:  # test modules are not an API surface
                continue
            try:
                # Bytes, not text: `read_text()` decodes with the locale
                # encoding, and CI runs Windows. Under cp1252 the em-dash in
                # the astro entry below comes back as "â€”", so the scan and
                # this file's own UTF-8 source would disagree about the same
                # docstring. `ast.parse` on bytes decodes the way the
                # interpreter does, PEP 263 cookie included.
                doc = ast.get_docstring(ast.parse(path.read_bytes()))
            except SyntaxError:  # pragma: no cover - not importable anyway
                continue
            if doc:
                out[doc.splitlines()[0].strip()].append(
                    path.relative_to(_ROOT).as_posix()
                )
    return out


def test_a_new_docstring_is_not_shared_across_directories() -> None:
    shared = {
        line: files
        for line, files in _first_lines().items()
        if len({pathlib.Path(f).parent for f in files}) > 1
        and line not in SHARED_BY_DESIGN
    }
    assert not shared, (
        "these module docstrings appear in more than one directory, which is how "
        "a copied file comes to describe the original's job. Rewrite the "
        "docstring to describe the file, or add the line to SHARED_BY_DESIGN "
        f"with a reason:\n{chr(10).join(f'  {k!r}: {v}' for k, v in shared.items())}"
    )


def test_the_allowlist_has_no_stale_entries() -> None:
    """An entry kept after its duplication is gone hides the next copy of it."""
    lines = _first_lines()
    stale = [
        line
        for line in SHARED_BY_DESIGN
        if len({pathlib.Path(f).parent for f in lines.get(line, [])}) < 2
    ]
    assert not stale, (
        f"SHARED_BY_DESIGN entries that are no longer shared: {stale}. "
        "Remove them, so a future copy of that docstring is caught."
    )
