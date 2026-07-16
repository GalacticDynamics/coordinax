"""Regression tests for hatch-vcs version command configuration."""

import tomllib
from pathlib import Path


def _read_pyproject(path: Path) -> dict:
    return tomllib.loads(path.read_text())


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
