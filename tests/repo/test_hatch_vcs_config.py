"""Regression tests for hatch-vcs version command configuration."""

import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

#: Every distribution in the workspace (root + the five sub-packages).
#:
#: Anchored to `__file__`, not the cwd. Relative paths made this silently
#: under-test: run from anywhere but the repo root the glob matched nothing,
#: the per-package parametrisation vanished, and the file still reported
#: success -- 14 cases collected from the root against 4 from `tests/`, with
#: `coordinaxs.curveframes` among the five lost. An empty parametrisation is
#: the quietest way for a guard to stop guarding.
_ALL_PYPROJECTS = [
    _ROOT / "pyproject.toml",
    *sorted((_ROOT / "packages").glob("coordinaxs.*/pyproject.toml")),
]


def _read_pyproject(path: Path) -> dict:
    return tomllib.loads(path.read_text())


@pytest.mark.parametrize("path", _ALL_PYPROJECTS, ids=lambda p: str(p.parent.name))
def test_license_metadata_is_pep639_consistent(path: Path) -> None:
    """No distribution mixes an SPDX ``license`` with a ``License ::`` classifier.

    PEP 639 makes the two mutually exclusive: a project that declares
    ``license = "MIT"`` (an SPDX expression, Metadata-Version 2.4
    ``License-Expression``) must not also carry a ``License :: ...`` trove
    classifier. Warehouse/PyPI rejects such an upload with HTTP 400, and
    ``twine check`` does not catch it — so this is guarded here instead.
    """
    project = _read_pyproject(path)["project"]
    has_spdx = isinstance(project.get("license"), str)
    license_classifiers = [
        c for c in project.get("classifiers", []) if c.startswith("License ::")
    ]

    if has_spdx:
        assert not license_classifiers, (
            f"{path} declares SPDX `license = {project['license']!r}` and also "
            f"carries {license_classifiers}; PEP 639 forbids both (PyPI rejects "
            "the upload). Drop the `License ::` classifier."
        )


@pytest.mark.parametrize("path", _ALL_PYPROJECTS, ids=lambda p: str(p.parent.name))
def test_distribution_ships_a_license_file(path: Path) -> None:
    """Every distribution has a LICENSE alongside its pyproject.

    Without a LICENSE file next to the pyproject, hatchling's default
    ``license-files`` glob finds nothing and the built wheel carries no license
    text — so the distribution ships without its license.
    """
    assert (path.parent / "LICENSE").is_file(), (
        f"{path.parent} has no LICENSE file; the built wheel would omit the "
        "license text."
    )


def test_main_package_uses_vcs_source() -> None:
    """Main package should use vcs as the version source."""
    cfg = _read_pyproject(_ROOT / "pyproject.toml")
    version_config = cfg["tool"]["hatch"]["version"]

    assert version_config.get("source") == "vcs"

    cmd = version_config["raw-options"]["scm"]["git"]["describe_command"]

    assert isinstance(cmd, list)
    assert cmd == [
        "git",
        "describe",
        "--dirty",
        "--tags",
        "--long",
        "--match",
        "coordinax-v*",
    ]


#: The sub-package distributions, enumerated rather than listed.
_PACKAGE_PYPROJECTS = sorted((_ROOT / "packages").glob("coordinaxs.*/pyproject.toml"))


@pytest.mark.parametrize("path", _PACKAGE_PYPROJECTS, ids=lambda p: str(p.parent.name))
def test_workspace_packages_use_package_specific_git_describe_match(path: Path) -> None:
    """Each sub-package matches only its own tags.

    The packages are read off disk and the expected pattern derived from the
    directory name, because the hardcoded pair of dicts this replaces listed
    four of the five: `coordinaxs.curveframes` was absent, so its match
    pattern was unguarded and a wrong one would have shipped silently. A
    listing that has to be updated by hand is a listing that eventually is
    not -- enumerating means the next package is covered by existing.

    The derivation (dots to dashes, then `-v*`) is checked against all five
    actual values, `coordinaxs.interop.astropy` -> `coordinaxs-interop-astropy-v*`
    included.
    """
    expected = f"{path.parent.name.replace('.', '-')}-v*"
    cmd = _read_pyproject(path)["tool"]["hatch"]["version"]["raw-options"]["scm"][
        "git"
    ]["describe_command"]

    assert isinstance(cmd, list)
    assert cmd == [
        "git",
        "describe",
        "--dirty",
        "--tags",
        "--long",
        "--match",
        expected,
    ]
