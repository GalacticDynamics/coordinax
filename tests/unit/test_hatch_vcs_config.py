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
        "coordinax.api": Path("packages/coordinax.api/pyproject.toml"),
        "coordinax.astro": Path("packages/coordinax.astro/pyproject.toml"),
        "coordinax.hypothesis": Path("packages/coordinax.hypothesis/pyproject.toml"),
        "coordinax.interop.astropy": Path(
            "packages/coordinax.interop.astropy/pyproject.toml"
        ),
    }

    expected_patterns = {
        "coordinax.api": "coordinax-api-v*",
        "coordinax.astro": "coordinax-astro-v*",
        "coordinax.hypothesis": "coordinax-hypothesis-v*",
        "coordinax.interop.astropy": "coordinax-interop-astropy-v*",
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
