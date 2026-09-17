r"""Report what this branch changes about `coordinax`'s public API.

**Reports; never fails.** A pull request is exactly where a deliberate API
change belongs, so gating on "the API changed" would block the normal case and
be overridden until it was ignored. What is wanted is to *notice* -- so this
writes a summary and exits 0. Pass ``--strict`` to exit non-zero instead, which
is for a release check, where the question is "does this need a major bump?"

**Multiple dispatch is excluded, and has to be.** `griffe` reads source
statically, and each ``@plum.dispatch def norm(...)`` rebinds the module-level
name, so a static reader keeps only the last: `charts/register_ptmap.py` writes
``def pt_map`` 35 times and griffe records one signature. Measured on that
module, it is wrong in both directions:

===============================  ==========  ==================================
change                           truth       griffe
===============================  ==========  ==================================
delete a non-last registration   breaking    0 reports (missed)
add a new registration           additive    7 x "Parameter was removed"
===============================  ==========  ==================================

The second is why this cannot simply be reported unfiltered: an additive PR
would be announced as removing every parameter, because the new method's
signature *replaces* the old one in griffe's view.

The exclusion is **derived at runtime** from `plum.Function` rather than
listed, because a written-down denylist would rot the first time a verb gained
or lost dispatch. What is left is most of the surface: of 182 public names 30
are `plum.Function`, and all 78 public methods on public classes are ordinary
functions.

Known gap, measured: griffe reports nothing for a changed *return* annotation,
so return-type narrowing is not covered. Parameter removals, parameter-kind
changes (positional-only to keyword-only) and removed public objects are.

See #117.
"""

__all__ = ("main",)

import argparse
import os
import pathlib
import subprocess
import sys

from typing import TYPE_CHECKING

import plum

if TYPE_CHECKING:
    import griffe

_PACKAGE = "coordinax"
_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Tried in order; the first that resolves is the comparison baseline.
_BASELINE_REFS = ("upstream/main", "origin/main", "main")


def _resolve_baseline(explicit: str | None) -> str | None:
    """Resolve the baseline: `explicit` if given, else the first `_BASELINE_REFS`.

    An explicit ref that does not resolve returns `None` rather than falling
    back. Falling back would compare against a *different* baseline than the
    caller asked for and report the difference as though it were theirs, which
    is worse than reporting nothing -- so the caller is told instead.
    """
    for ref in (explicit,) if explicit else _BASELINE_REFS:
        done = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],  # noqa: S607
            cwd=_ROOT,
            capture_output=True,
            check=False,
        )
        if done.returncode == 0:
            return ref
    return None


def dispatched_paths() -> frozenset[str]:
    """Collect fully-qualified paths of every `plum.Function` in the package.

    Read from the live objects, not the source: dispatch exists only at
    runtime, which is the whole reason griffe cannot see it.

    Qualified, not bare. Bare names look sufficient and are not: eight modules
    here are *named* after a dispatched verb -- `norm.py`, `interval.py`,
    `add.py`, `angle_between.py`, `chord_distance.py`, `geodesic_distance.py`,
    `scale_factors.py`, `tangent_map.py` -- so matching a bare "norm" anywhere
    in a dotted path silently excluded every object defined in those files,
    dispatch or not, along with any class method sharing a verb's name
    (`Coordinate.cconvert`). Excluding real breakages is the one failure this
    script cannot afford, since its output is trusted precisely where nobody
    is looking closely.
    """
    import coordinax as cx  # noqa: F401, PLC0415  (populates `sys.modules`)

    paths: set[str] = set()
    for mod_name, module in list(sys.modules.items()):
        if not mod_name.startswith(_PACKAGE) or module is None:
            continue
        paths |= {
            f"{mod_name}.{attr}"
            for attr, val in vars(module).items()
            if isinstance(val, plum.Function)
        }
    return frozenset(paths)


def _is_dispatched(path: str, excluded: frozenset[str]) -> bool:
    """Report whether `path` is a dispatched function, or inside one.

    Exact match, or a descendant of one -- never a bare-name match, which
    collides with same-named modules and methods.
    """
    return path in excluded or any(path.startswith(f"{p}.") for p in excluded)


def find_changes(ref: str) -> "list[griffe.Breakage]":
    """Find breaking changes against `ref`, less the dispatched surface.

    `griffe` is imported here, not at module scope, so the filter above stays
    importable without it. The test suite exercises `_is_dispatched` in jobs
    that install only the `test` group, and a module-level import made the
    whole file unimportable there -- which broke collection rather than
    skipping, since the filter itself needs no griffe at all.
    """
    import griffe  # noqa: PLC0415  (see above)

    baseline = griffe.load_git(
        _PACKAGE, ref=ref, repo=_ROOT, search_paths=["src"], allow_inspection=False
    )
    current = griffe.load(
        _PACKAGE, search_paths=[_ROOT / "src"], allow_inspection=False
    )
    excluded = dispatched_paths()
    if not any(p.endswith(".pt_map") for p in excluded):
        msg = (
            "`pt_map` is the canonical dispatched verb; if it is no longer a "
            "`plum.Function`, this exclusion has stopped excluding anything "
            "and the report below would be mostly noise"
        )
        raise RuntimeError(msg)
    return [
        change
        for change in griffe.find_breaking_changes(baseline, current)
        if not _is_dispatched(change.obj.path, excluded)
    ]


def render(ref: str, changes: "list[griffe.Breakage]") -> str:
    """Render a Markdown summary, for `$GITHUB_STEP_SUMMARY` or a terminal."""
    if not changes:
        return (
            f"### Public API\n\nNo breaking changes against `{ref}` "
            "(excluding the dispatched surface, which griffe cannot read).\n"
        )
    rows = "\n".join(
        f"| `{c.obj.path}` | {c.kind.value} |"
        for c in sorted(changes, key=lambda c: c.obj.path)
    )
    return (
        f"### Public API: {len(changes)} breaking change(s) against `{ref}`\n\n"
        "This is **not** a failure -- a pull request is where a deliberate API\n"
        "change belongs. It is here so the change is noticed and the release\n"
        "notes and version bump can account for it.\n\n"
        f"| object | change |\n| --- | --- |\n{rows}\n\n"
        "Dispatched (`plum.Function`) names are excluded: griffe sees only the\n"
        "last registration of each, so it both misses real removals and reports\n"
        "additions as removals. See this script's docstring.\n"
    )


def _emit(report: str) -> None:
    """Print the report, and append it to the CI job summary when there is one.

    The summary write is best-effort. stdout is the reliable channel; a
    summary file that is missing or unwritable is a worse outcome if it takes
    the whole run down with it, since this job is not supposed to fail.
    """
    print(report)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        try:
            with pathlib.Path(summary).open("a", encoding="utf-8") as fh:
                fh.write(report)
        except OSError as exc:  # pragma: no cover - CI filesystem only
            print(f"(could not write the job summary: {exc})")


def _run(args: argparse.Namespace) -> int:
    """Produce the report for already-parsed `args`."""
    ref = _resolve_baseline(args.ref)
    if ref is None:
        tried = args.ref or ", ".join(_BASELINE_REFS)
        # Written to the summary too, not just stdout: a missing baseline used
        # to leave the job green with an empty summary, which reads exactly
        # like "no API changes" and is the one outcome this must never fake.
        _emit(
            "### Public API\n\n**No report.** No baseline ref resolved "
            f"(tried `{tried}`). Nothing was compared -- this is *not* a "
            "statement that the API is unchanged. A shallow checkout is the "
            "usual cause; this job needs `fetch-depth: 0`.\n"
        )
        return 1 if args.strict else 0

    changes = find_changes(ref)
    _emit(render(ref, changes))
    return 1 if (args.strict and changes) else 0


def main(argv: list[str] | None = None) -> int:
    """Write the report; return 0 unless `--strict` and something went wrong.

    Every failure is *reported*, not raised. An exception escaping here exits
    non-zero, reddens the job, and -- if the check were ever added to a
    required list -- turns the report into the gate it is written not to be.
    So the whole run is guarded, not just the comparison: resolving the
    baseline shells out to git, and writing the summary touches the
    filesystem, and neither is worth failing a non-gating job over. The reason
    goes to the summary, where it is visible instead of buried in a log, and
    `--strict` still returns 1 so a release check cannot pass on a report that
    never ran.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default=None, help="baseline ref to compare against")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when the API broke (for a release check)",
    )
    args = parser.parse_args(argv)

    try:
        return _run(args)
    except Exception as exc:  # noqa: BLE001  (see the docstring)
        _emit(
            "### Public API\n\n**No report.** The run did not complete: "
            f"`{type(exc).__name__}: {exc}`\n\nNothing was compared -- this "
            "is *not* a statement that the API is unchanged.\n"
        )
        return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
