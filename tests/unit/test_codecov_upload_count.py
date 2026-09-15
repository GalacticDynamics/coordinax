"""`codecov.yml`'s ``after_n_builds`` must match what CI actually uploads.

That file waits for every upload before scoring a commit, and says why: judged
early, "every line only [`check_oldest`] covers reads as a miss: #794 scored
-5.35% (-540 hits) on a diff that deleted 41 lines and added no uncovered
ones, and `codecov/project` failed on a PR whose coverage was fine."

The count is hardcoded, and its own comment names the hazard: "Adding a version
there without bumping it here brings the early verdict back; removing one
leaves Codecov waiting for an upload that never comes." It defers wiring the
value to the matrix until "that drift bites" -- so this does not wire it, it
just fails when the two disagree.
"""

__all__: tuple[str, ...] = ()

import json
import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CI = (_ROOT / ".github/workflows/ci.yml").read_text()
_CODECOV = (_ROOT / "codecov.yml").read_text()


def _search(pattern: str, text: str, what: str) -> re.Match[str]:
    """Match *pattern*, failing loudly rather than vacuously passing."""
    match = re.search(pattern, text, re.MULTILINE)
    assert match is not None, f"could not find {what}; this guard needs updating"
    return match


def test_after_n_builds_matches_the_jobs_that_upload() -> None:
    versions = json.loads(
        _search(r"^\s*PYTHON_VERSIONS:\s*'(\[.*\])'", _CI, "PYTHON_VERSIONS").group(1)
    )
    nonlinux_os = _search(
        r"^\s*runs-on:\s*\[([^\]]+)\]", _CI, "the tests_nonlinux OS list"
    ).group(1)
    n_nonlinux_os = len([o for o in nonlinux_os.split(",") if o.strip()])

    # tests_linux runs every version; tests_nonlinux runs the edge version
    # (`python-edge` is `versions[-1]`) on each OS; check_oldest runs once.
    expected = len(versions) + n_nonlinux_os + 1

    declared = [
        int(m)
        for m in re.findall(r"^\s*after_n_builds:\s*(\d+)", _CODECOV, re.MULTILINE)
    ]
    assert declared, "codecov.yml declares no after_n_builds"
    assert set(declared) == {expected}, (
        f"codecov.yml waits for {declared} uploads, CI produces {expected} "
        f"({len(versions)} linux + {n_nonlinux_os} non-linux + 1 oldest)"
    )


def test_only_the_three_known_jobs_upload_coverage() -> None:
    """A new uploading job changes the count, so it has to be noticed."""
    uploads = len(re.findall(r"codecov/codecov-action", _CI))
    assert uploads == 3, (
        f"{uploads} codecov upload steps in ci.yml, expected 3 "
        "(tests_linux, tests_nonlinux, check_oldest) -- update the arithmetic "
        "in test_after_n_builds_matches_the_jobs_that_upload"
    )
