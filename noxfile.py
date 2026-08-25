#!/usr/bin/env -S uv run --script  # noqa: EXE001
# /// script
#    dependencies = ["nox", "nox_uv"]
# ///
"""Nox setup."""

import argparse
import shutil
import tomllib
from enum import Enum
from pathlib import Path

from typing import Self, final

import nox
from nox_uv import session

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv"

DIR = Path(__file__).parent.resolve()


class _StrEnumWithPaths(str, Enum):
    """String enum that carries an immutable tuple of paths."""

    _paths: tuple[str, ...]

    def __new__(cls, value: str, paths: tuple[str, ...]) -> Self:
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._paths = paths
        return obj

    def __str__(self) -> str:
        """Match StrEnum behavior: render as the raw string value."""
        return self.value

    @property
    def paths(self) -> tuple[str, ...]:
        """Get paths attached to this enum value."""
        return self._paths


@final
class PackageEnum(_StrEnumWithPaths):
    """Enum for package names."""

    def __repr__(self) -> str:
        return f"{self.value!r}"

    coordinax = ("coordinax", ("README.md", "docs", "skills/", "src/", "tests/"))
    api = ("api", ("packages/coordinaxs.api/",))
    astro = ("astro", ("packages/coordinaxs.astro/",))
    curveframes = ("curveframes", ("packages/coordinaxs.curveframes/",))
    hypothesis = ("hypothesis", ("packages/coordinaxs.hypothesis/",))


# =============================================================================
# Comprehensive sessions


@session(
    uv_groups=["lint", "test", "docs"],
    uv_extras=["workspace"],
    reuse_venv=True,
    default=True,
)
def all(s: nox.Session, /) -> None:  # noqa: A001
    """Run all default sessions."""
    s.notify("lint")
    s.notify("test")
    s.notify("docs")


# =============================================================================
# Linting


@session(uv_groups=["lint"], reuse_venv=True)
def lint(s: nox.Session, /) -> None:
    """Run the linter."""
    s.notify("precommit")
    # s.notify("pylint") # TODO: re-enable after fixing lint errors
    s.notify("ty")


@session(uv_groups=["lint"], uv_extras=["workspace"], reuse_venv=True)
def precommit(s: nox.Session, /) -> None:
    """Run the linter."""
    s.run("prek", "run", "--all-files", *s.posargs)


@session(uv_groups=["lint"], reuse_venv=True)
@nox.parametrize("package", list(PackageEnum))
def pylint(s: nox.Session, /, package: PackageEnum) -> None:
    """Run PyLint."""
    package_paths = (
        ("src/coordinax",) if package == PackageEnum.coordinax else tuple(package.paths)
    )
    s.run("pylint", *package_paths, *s.posargs)


@session(uv_groups=["lint"], reuse_venv=True)
@nox.parametrize("package", list(PackageEnum))
def ty(s: nox.Session, /, package: PackageEnum) -> None:
    """Run ty."""
    package_paths = (
        ("src/coordinax", "packages/coordinaxs.api/")
        if package == PackageEnum.coordinax
        else tuple(package.paths)
    )
    s.run("ty", "check", *package_paths, *s.posargs)


# =============================================================================
# Testing


@session(uv_groups=["test"], uv_extras=["workspace"], reuse_venv=True, default=True)
def test(s: nox.Session, /) -> None:
    """Run the unit and regular tests.

    Optional flags:
      --exclude-package <package>
          Exclude one package's paths from this aggregated pytest invocation.
          May be provided multiple times.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--exclude-package", action="append", default=[])
    args, posargs = parser.parse_known_args(s.posargs)

    excluded: set[str] = set()
    for raw in args.exclude_package:
        try:
            package = PackageEnum[raw]
        except KeyError:
            s.error(f"Unknown --exclude-package value: {raw!r}")
        excluded.update(path.rstrip("/") for path in package.paths)

    # Select by naming the paths that survive rather than by `--ignore`.
    # `testpaths` lists these package directories explicitly, and pytest does
    # not apply `--ignore` to an initial argument it was handed directly, so
    # every `--ignore` here was a no-op: the jobs that asked to skip a package
    # still ran the whole suite.
    path_args: list[str] = []
    if excluded:
        cfg = tomllib.loads((DIR / "pyproject.toml").read_text(encoding="utf-8"))
        testpaths: list[str] = cfg["tool"]["pytest"]["ini_options"]["testpaths"]
        path_args = [p for p in testpaths if p.rstrip("/") not in excluded]
        # Excluding nothing is never what the caller meant, and quietly running
        # everything is how this went unnoticed. Fail instead.
        if len(path_args) == len(testpaths):
            s.error(
                f"--exclude-package matched no testpaths: {sorted(excluded)} "
                f"against {testpaths}"
            )

    # -n logical: parallelize across cores. --dist=loadfile: keep each file's
    # tests on one worker -- Sybil doctests share sequential state across
    # `>>>` examples within a source file, which breaks if xdist scatters
    # them across workers. xdist breaks --pdb/--trace, so skip it when either
    # is requested rather than relying on the caller to also pass `-n0`.
    debugging = any(arg == "--trace" or arg.startswith("--pdb") for arg in posargs)
    xdist_args = [] if debugging else ["-n", "logical", "--dist=loadfile"]

    # This session installs the `workspace` extra (interop included), so the
    # interop order-independence tests must run, not silently skip.
    s.run(
        "pytest",
        *xdist_args,
        *path_args,
        *posargs,
        env={"COORDINAX_REQUIRE_INTEROP_TESTS": "1"},
    )
    # s.notify("pytest_benchmark", posargs=s.posargs)


@session(uv_groups=["test"], uv_extras=["workspace"], reuse_venv=True)
@nox.parametrize("package", list(PackageEnum))
def pytest(s: nox.Session, /, package: PackageEnum) -> None:
    """Run the unit and regular tests."""
    s.run("pytest", *package.paths, *s.posargs)


# =============================================================================
# Documentation


@session(uv_groups=["docs"], uv_extras=["workspace"], reuse_venv=True)
def docs(s: nox.Session, /) -> None:
    """Build the docs. Pass "--serve" to serve. Pass "-b linkcheck" to check links."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    parser.add_argument("--output-dir", dest="output_dir", default="_build")
    args, posargs = parser.parse_known_args(s.posargs)

    if args.builder != "html" and args.serve:
        s.error("Must not specify non-HTML builder with --serve")

    s.chdir("docs")

    # Convert jupytext markdown files to notebooks
    s.run(
        "jupytext",
        "--to",
        "notebook",
        "guides/perf.md",
        "--output",
        "guides/perf.ipynb",
    )
    s.run(
        "jupytext",
        "--to",
        "notebook",
        "packages/coordinaxs.curveframes/visualizing.md",
        "--output",
        "packages/coordinaxs.curveframes/visualizing.ipynb",
    )

    if args.builder == "linkcheck":
        s.run("sphinx-build", "-b", "linkcheck", ".", "_build/linkcheck", *posargs)
        return

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        "-W",  # turn warnings into errors
        f"-b={args.builder}",
        f"-d {args.output_dir}/doctrees",
        "-D",
        "language=en",
        ".",
        f"{args.output_dir}/{args.builder}",
        *posargs,
    )

    if args.serve:
        s.run("sphinx-autobuild", *shared_args)
    else:
        s.run("sphinx-build", "--keep-going", *shared_args)


@session(uv_groups=["docs"], reuse_venv=True)
def build_api_docs(s: nox.Session, /) -> None:
    """Build (regenerate) API docs."""
    s.chdir("docs")
    s.run(
        "sphinx-apidoc",
        "-o",
        "api/",
        "--module-first",
        "--no-toc",
        "--force",
        "../src/coordinax",
    )


# =============================================================================
# Packaging


@session(uv_groups=["build"])
def build(s: nox.Session, /) -> None:
    """Build an SDist and wheel."""
    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)
    s.run("python", "-m", "build")


################################################################################

if __name__ == "__main__":
    nox.main()
