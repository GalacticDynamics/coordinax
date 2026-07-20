"""Regression tests for hatch-vcs version command configuration."""

import tomllib
from pathlib import Path

import pytest

#: Every distribution in the workspace (root + the five sub-packages).
_ALL_PYPROJECTS = [
    Path("pyproject.toml"),
    *sorted(Path("packages").glob("coordinaxs.*/pyproject.toml")),
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
    cfg = _read_pyproject(Path("pyproject.toml"))
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


def test_workspace_packages_use_package_specific_git_describe_match() -> None:
    """Workspace packages use git describe with package match patterns."""
    package_patterns = {
        "coordinaxs.api": Path("packages/coordinaxs.api/pyproject.toml"),
        "coordinaxs.astro": Path("packages/coordinaxs.astro/pyproject.toml"),
        "coordinaxs.hypothesis": Path("packages/coordinaxs.hypothesis/pyproject.toml"),
        "coordinaxs.interop.astropy": Path(
            "packages/coordinaxs.interop.astropy/pyproject.toml"
        ),
    }

    expected_patterns = {
        "coordinaxs.api": "coordinaxs-api-v*",
        "coordinaxs.astro": "coordinaxs-astro-v*",
        "coordinaxs.hypothesis": "coordinaxs-hypothesis-v*",
        "coordinaxs.interop.astropy": "coordinaxs-interop-astropy-v*",
    }

    for package, path in package_patterns.items():
        cfg = _read_pyproject(path)
        cmd = cfg["tool"]["hatch"]["version"]["raw-options"]["scm"]["git"][
            "describe_command"
        ]

        assert isinstance(cmd, list)
        assert cmd == [
            "git",
            "describe",
            "--dirty",
            "--tags",
            "--long",
            "--match",
            expected_patterns[package],
        ]
