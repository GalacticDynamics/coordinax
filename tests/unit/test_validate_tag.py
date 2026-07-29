"""Unit tests for scripts/validate_tag.py.

The script answers two questions, and the tests are the two tables that go with
them:

* `parse_version_tag(tag)` -- a pure string parse, so a (tag, expected) table.
* `validate_tag_for_package(tag, package)` -- the release rules, so a
  (tag, package, valid, error fragments) table. Rows that reach git are split
  out, because those need a canned `git tag -l` result.

The rules under test:

- package-specific tags required (not bare vX.Y.Z), from 0.24 on
- .0 releases require a matching coordinator tag
- legacy (<0.24) exceptions
- subprocess-based coordinator lookup, with error handling
"""

__all__: tuple[str, ...] = ()

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def validate_tag(monkeypatch):
    """Import validate_tag module with temporary path modification.

    Uses monkeypatch.syspath_prepend() to avoid permanent interpreter mutation.
    Path change is automatically reverted after the test completes.
    """
    scripts_path = Path(__file__).parent.parent.parent / "scripts"
    monkeypatch.syspath_prepend(str(scripts_path))

    # Import inside fixture so path change is scoped
    import validate_tag as vt

    return vt


@pytest.fixture
def fake_git():
    """Patch `subprocess.run` with a canned `git tag -l` result.

    The twelve call sites that needed one previously built the same four-line
    `MagicMock` by hand.
    """

    def _patch(stdout: str = "", *, returncode: int = 0, stderr: str = ""):
        result = MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)
        return patch("subprocess.run", return_value=result)

    return _patch


# ===================================================================
# parse_version_tag


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        # package-specific
        ("coordinax-v1.0.0", ("coordinax", 1, 0, 0)),
        ("coordinaxs-api-v2.3.4", ("coordinaxs-api", 2, 3, 4)),
        ("coordinaxs-astro-v0.5.2", ("coordinaxs-astro", 0, 5, 2)),
        ("coordinaxs-interop-astropy-v1.2.3", ("coordinaxs-interop-astropy", 1, 2, 3)),
        # bare coordinator tags -- empty package name
        ("v1.0.0", ("", 1, 0, 0)),
        ("v2.3.4", ("", 2, 3, 4)),
        ("v0.24.0", ("", 0, 24, 0)),
        # rejected
        ("1.0.0", None),  # missing 'v' prefix
        ("va.b.c", None),  # non-numeric version
        ("coordinax-1.0.0", None),  # missing 'v'
        ("coordinax_v1.0.0", None),  # underscore instead of dash
        ("v1.0", None),  # missing patch version
        ("random-tag", None),  # not a version tag
    ],
)
def test_parse_version_tag(validate_tag, tag: str, expected) -> None:
    assert validate_tag.parse_version_tag(tag) == expected


# ===================================================================
# check_coordinator_tag_exists


@pytest.mark.parametrize(
    ("stdout", "exists"),
    [
        ("v1.0.0\n", True),
        ("\n", False),
    ],
    ids=["found", "absent"],
)
def test_check_coordinator_tag_exists(validate_tag, fake_git, stdout, exists) -> None:
    with fake_git(stdout):
        assert validate_tag.check_coordinator_tag_exists("1.0.0") is exists


@pytest.mark.parametrize(
    ("returncode", "stderr"),
    [(128, "fatal: not a git repository"), (1, "")],
    ids=["with-stderr", "without-stderr"],
)
def test_git_failure_raises(validate_tag, fake_git, returncode, stderr) -> None:
    """A non-zero `git tag -l` is an error, not an absent tag."""
    with (
        fake_git("", returncode=returncode, stderr=stderr),
        pytest.raises(RuntimeError, match="git tag -l failed"),
    ):
        validate_tag.check_coordinator_tag_exists("1.0.0")


# ===================================================================
# validate_tag_for_package -- rules that never reach git

#: (tag, package, valid, fragments the error message must contain)
OFFLINE_CASES = [
    # legacy: <= 0.23 accepts bare *and* package-specific tags
    ("v0.23.0", "coordinax", True, ()),
    ("v0.20.5", "coordinax", True, ()),
    ("v0.23.10", "coordinaxs.api", True, ()),
    ("coordinax-v0.23.0", "coordinax", True, ()),
    # unparsable
    ("invalid-tag", "coordinax", False, ("Invalid tag format",)),
    ("1.0.0", "coordinax", False, ("Invalid tag format",)),
    # bare tags rejected from 0.24 on
    ("v0.24.0", "coordinax", False, ("Coordinator tags", "package-specific tags")),
    ("v1.0.0", "coordinaxs.api", False, ("Coordinator tags",)),
    # the tag's package must match the workflow's
    (
        "coordinaxs-api-v0.24.0",
        "coordinax",
        False,
        ("This tag is for package 'coordinaxs.api'", "this workflow is for package"),
    ),
    (
        "coordinax-v0.24.0",
        "coordinaxs.api",
        False,
        ("This tag is for package 'coordinax'",),
    ),
    # unknown package
    (
        "invalid-package-v1.0.0",
        "invalid-package",
        False,
        ("Unknown package", "Allowed values"),
    ),
    # bugfix releases (.1+) need no coordinator tag, so never reach git
    ("coordinax-v0.24.1", "coordinax", True, ()),
    ("coordinaxs-api-v1.5.3", "coordinaxs.api", True, ()),
    ("coordinaxs-astro-v2.0.99", "coordinaxs.astro", True, ()),
    ("coordinaxs-curveframes-v2.0.99", "coordinaxs.curveframes", True, ()),
    # package=None defaults to coordinax
    (
        "coordinaxs-api-v0.24.0",
        None,
        False,
        ("This tag is for package 'coordinaxs.api'", "this workflow is for package"),
    ),
]


@pytest.mark.parametrize(
    ("tag", "package", "valid", "fragments"),
    OFFLINE_CASES,
    ids=[f"{tag}-for-{package}" for tag, package, _, _ in OFFLINE_CASES],
)
def test_validate_tag_without_git(validate_tag, tag, package, valid, fragments) -> None:
    """These rules resolve without git, and `subprocess.run` proves it.

    Patched rather than left alone: if a regression made one of these paths
    shell out, an unpatched test would run a real `git tag -l` and pass or fail
    on whatever tags the CI checkout happens to have. Patching turns that into
    a deterministic failure here.
    """
    with patch("subprocess.run") as run:
        is_valid, error = validate_tag.validate_tag_for_package(tag, package)
    run.assert_not_called()
    assert is_valid is valid, error
    if valid:
        assert error == ""
    for fragment in fragments:
        assert fragment in error


# ===================================================================
# validate_tag_for_package -- .0 releases, which consult git

#: (tag, package, `git tag -l` output, valid, error fragments)
DOT_ZERO_CASES = [
    ("coordinax-v0.24.0", "coordinax", "v0.24.0\n", True, ()),
    ("coordinaxs-api-v1.0.0", "coordinaxs.api", "v1.0.0\n", True, ()),
    ("coordinax-v0.24.0", None, "v0.24.0\n", True, ()),
    # large version numbers are not special
    ("coordinax-v99.999.999", "coordinax", "v99.999.999\n", True, ()),
    # the match is per-line, so extra tags and surrounding whitespace are fine
    ("coordinax-v1.0.0", "coordinax", "v1.0.0\nv1.0.0-rc1\n", True, ()),
    ("coordinax-v1.0.0", "coordinax", "  v1.0.0  \n", True, ()),
    # no coordinator tag -> rejected
    (
        "coordinax-v0.24.0",
        "coordinax",
        "\n",
        False,
        ("must have a corresponding coordinator tag", "v0.24.0"),
    ),
]


@pytest.mark.parametrize(
    ("tag", "package", "git_stdout", "valid", "fragments"),
    DOT_ZERO_CASES,
    ids=[f"{tag}-{'found' if v else 'missing'}" for tag, _, _, v, _ in DOT_ZERO_CASES],
)
def test_validate_dot_zero_release(
    validate_tag, fake_git, tag, package, git_stdout, valid, fragments
) -> None:
    with fake_git(git_stdout):
        is_valid, error = validate_tag.validate_tag_for_package(tag, package)
    assert is_valid is valid, error
    if valid:
        assert error == ""
    for fragment in fragments:
        assert fragment in error


def test_every_release_package_accepts_its_own_tag(validate_tag, fake_git) -> None:
    """Every name in PACKAGE_NAMES validates against its own vX.0.0 tag."""
    with fake_git("v1.0.0\n"):
        for package in validate_tag.PACKAGE_NAMES:
            tag = f"{package.replace('.', '-')}-v1.0.0"
            is_valid, error = validate_tag.validate_tag_for_package(tag, package)
            assert is_valid is True, f"{package}: {error}"
            assert error == ""


def test_curveframes_is_a_release_package(validate_tag) -> None:
    """Regression: curveframes must not fall out of the release set."""
    assert "coordinaxs.curveframes" in validate_tag.PACKAGE_NAMES


# ===================================================================
# When git is consulted, and how


def test_dot_zero_release_queries_the_exact_coordinator_tag(
    validate_tag, fake_git
) -> None:
    with fake_git("v0.24.0\n") as run:
        is_valid, _ = validate_tag.validate_tag_for_package(
            "coordinax-v0.24.0", "coordinax"
        )
    assert is_valid is True
    run.assert_called_once()
    assert run.call_args[0][0] == ["git", "tag", "-l", "v0.24.0"]


@pytest.mark.parametrize(
    "tag", ["coordinax-v0.24.1", "coordinax-v0.24.5"], ids=["patch-1", "patch-5"]
)
def test_bugfix_release_never_shells_out(validate_tag, tag) -> None:
    with patch("subprocess.run") as run:
        is_valid, _ = validate_tag.validate_tag_for_package(tag, "coordinax")
    assert is_valid is True
    run.assert_not_called()


@pytest.mark.parametrize(
    ("returncode", "stderr"),
    [(128, "fatal: not a git repository"), (1, "Failed to fetch tags")],
    ids=["not-a-repo", "fetch-failed"],
)
def test_git_failure_propagates_out_of_validation(
    validate_tag, fake_git, returncode, stderr
) -> None:
    """A git failure during a .0 check surfaces, rather than reading as absent."""
    with (
        fake_git("", returncode=returncode, stderr=stderr),
        pytest.raises(RuntimeError) as exc_info,
    ):
        validate_tag.validate_tag_for_package("coordinax-v0.24.0", "coordinax")

    message = str(exc_info.value)
    assert "git tag -l failed" in message
    assert "fetch-depth" in message


# ===================================================================
# Boundary


@pytest.mark.parametrize(
    ("tag", "valid"),
    [("v0.23.0", True), ("v0.24.0", False)],
    ids=["legacy-0.23", "modern-0.24"],
)
def test_bare_tag_boundary_at_0_24(validate_tag, tag, valid) -> None:
    """0.23 is the last version where a bare coordinator tag is accepted."""
    is_valid, _ = validate_tag.validate_tag_for_package(tag, "coordinax")
    assert is_valid is valid
