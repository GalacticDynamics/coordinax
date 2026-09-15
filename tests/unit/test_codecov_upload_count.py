"""`codecov.yml`'s ``after_n_builds`` must match what CI actually uploads.

That file waits for every upload before scoring, because judged early the
lines only `check_oldest` covers read as misses -- #794 scored -5.35% on a
diff that added no uncovered lines. The count is hardcoded, and its own
comment names the hazard: "Adding a version there without bumping it here
brings the early verdict back; removing one leaves Codecov waiting for an
upload that never comes." That comment defers wiring the value to the matrix
until "the drift bites", so this does not wire it -- it just fails when the
two disagree.
"""

__all__: tuple[str, ...] = ()

import json
import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CI = (_ROOT / ".github/workflows/ci.yml").read_text()
_CODECOV = (_ROOT / "codecov.yml").read_text()


def _search(pattern: str, text: str, what: str) -> re.Match[str]:
    match = re.search(pattern, text, re.MULTILINE)
    assert match is not None, f"could not find {what}; this guard needs updating"
    return match


def test_after_n_builds_matches_the_jobs_that_upload() -> None:
    versions = json.loads(
        _search(r"^\s*PYTHON_VERSIONS:\s*'(\[.*\])'", _CI, "PYTHON_VERSIONS").group(1)
    )
    n_os = len(
        _search(r"^\s*runs-on:\s*\[([^\]]+)\]", _CI, "the tests_nonlinux OS list")
        .group(1)
        .split(",")
    )

    # tests_linux runs every version; tests_nonlinux runs the edge version
    # (`python-edge` is `versions[-1]`) on each OS; check_oldest runs once.
    expected = len(versions) + n_os + 1

    declared = set(re.findall(r"^\s*after_n_builds:\s*(\d+)", _CODECOV, re.MULTILINE))
    assert declared == {str(expected)}, (
        f"codecov.yml waits for {declared or 'no'} uploads, CI produces {expected} "
        f"({len(versions)} linux + {n_os} non-linux + 1 oldest)"
    )


def test_only_the_three_known_jobs_upload_coverage() -> None:
    """A new uploading job changes the count, so it has to be noticed."""
    uploads = len(re.findall(r"codecov/codecov-action", _CI))
    assert uploads == 3, (
        f"{uploads} codecov upload steps in ci.yml, expected 3 "
        "(tests_linux, tests_nonlinux, check_oldest)"
    )
