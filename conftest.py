"""Doctest configuration."""

import importlib
import pkgutil
import sys
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
from pathlib import Path

from collections.abc import Callable, Iterable, Sequence
from types import ModuleType

import pytest
from sybil import Document, Region, Sybil, document as sybil_document
from sybil.evaluators.doctest import DocTestEvaluator
from sybil.parsers import myst, rest
from sybil.parsers.abstract.doctest import DocTestStringParser
from sybil.python import import_path as sybil_import_path

optionflags = ELLIPSIS | NORMALIZE_WHITESPACE

# =========================================================
# Paths and Namespaces

ModuleRoot = tuple[Path, tuple[str, ...]]
COORDINAX_NAMESPACE: tuple[str, ...] = ("coordinax",)

CX_WORKSPACE_ROOT = Path(__file__).parent
CX_PACKAGES_ROOT = CX_WORKSPACE_ROOT / "packages"


def _discover_module_roots() -> tuple[ModuleRoot, ...]:
    """Discover all workspace ``src/coordinax`` roots.

    Every source tree rooted at ``*/src/coordinax`` maps to the same canonical
    Python namespace prefix ``coordinax``.
    """
    roots: list[ModuleRoot] = []

    main_root = CX_WORKSPACE_ROOT / "src" / "coordinax"
    if main_root.exists():
        roots.append((main_root, COORDINAX_NAMESPACE))

    if CX_PACKAGES_ROOT.exists():
        for package_dir in sorted(CX_PACKAGES_ROOT.iterdir()):
            package_root = package_dir / "src" / "coordinax"
            if package_root.exists():
                roots.append((package_root, COORDINAX_NAMESPACE))

    return tuple(roots)


MODULE_ROOTS: tuple[ModuleRoot, ...] = _discover_module_roots()
RESOLVED_MODULE_ROOTS: tuple[ModuleRoot, ...] = tuple(
    (root.resolve(), namespace) for root, namespace in MODULE_ROOTS
)


# =========================================================
# Module Alias Normalization


def _path_to_module(path: Path, root: Path, namespace: tuple[str, ...], /) -> str:
    """Map a Python file under ``root`` to a fully-qualified module path."""
    relative = path.relative_to(root)
    if relative.name == "__init__.py":
        suffix = tuple(relative.parts[:-1])
    else:
        suffix = (*tuple(relative.parts[:-1]), relative.stem)
    return ".".join((*namespace, *suffix))


def _module_is_in_workspace_roots(module: ModuleType, /) -> bool:
    """Return ``True`` if a module file lives under any discovered module root."""
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False

    resolved = Path(module_file).resolve()
    return any(resolved.is_relative_to(root) for root, _ in RESOLVED_MODULE_ROOTS)


def _normalize_namespace_aliases(package_name: str, /) -> None:
    """Normalize short module aliases for all submodules under ``package_name``.

    During mixed source-path and package-path test collection, modules can be
    loaded under both canonical names (e.g. ``coordinax.charts._src.d1``) and
    short aliases (e.g. ``charts._src.d1``). This normalization maps any
    workspace-local short alias back to the canonical loaded module object.
    """
    package = importlib.import_module(package_name)

    # Preload the package tree so canonical module objects exist.
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module_info.name)

    prefix = package_name + "."
    # Snapshot `sys.modules` because we mutate it in this loop.
    for name, module in list(sys.modules.items()):
        if not name.startswith(prefix):
            continue

        alias = name.removeprefix(prefix)
        existing = sys.modules.get(alias)

        # Mixed collection can import the same file as both
        # ``coordinax.foo`` and ``foo``. Make aliases point to the canonical
        # package module, but avoid clobbering unrelated third-party modules.
        if existing is None or _module_is_in_workspace_roots(existing):
            sys.modules[alias] = module


def pytest_configure(config: pytest.Config) -> None:
    """Normalize module aliases before collection for consistent identities.

    This repository intentionally runs mixed pytest collection targets,
    including both source paths (for Sybil/doctest-style collection) and normal
    package imports from tests. In that mode, Python can load the same file
    twice under different module names, for example:

    - ``coordinax.charts._src.d1`` (canonical package import)
    - ``charts._src.d1`` (path-derived/top-level alias)

    Those two module objects define distinct class objects, even though they
    come from the same file. That breaks identity-sensitive behavior in tests
    and dispatch, such as strict ``type(self) is type(other)`` equality checks
    and cached/multi-dispatch registries.

    At pytest startup (before collection), we normalize short aliases to point
    at already-imported canonical ``coordinax.*`` modules via
    :func:`_normalize_namespace_aliases`.

    Resulting guarantees:

    - Workspace modules have one effective runtime identity.
    - Source-path and package-path collection interoperate safely.
    - Chart equality/dispatch behavior remains stable across collection modes.

    """
    _normalize_namespace_aliases("coordinax")


# =========================================================
# Sybil Import Hook


def _import_path_with_namespace(path: Path, /) -> ModuleType:
    """Import workspace package files via canonical module names.

    Sybil receives filesystem paths. Without this mapping, those files may be
    imported as top-level modules (e.g. ``charts._src``), causing duplicate
    module identities versus ``coordinax.charts._src``.
    """
    resolved_path = path.resolve()
    for root, namespace in RESOLVED_MODULE_ROOTS:
        if resolved_path.is_relative_to(root):
            return importlib.import_module(
                _path_to_module(resolved_path, root, namespace)
            )

    return sybil_import_path(path)


sybil_document.import_path = _import_path_with_namespace


# =========================================================
# Sybil Parser Setup


class PyconCodeBlockParser:
    """Parser for MyST pycon code blocks with doctest evaluation."""

    def __init__(self, doctest_optionflags: int = 0) -> None:
        """Initialize the pycon code block parser.

        Parameters
        ----------
        doctest_optionflags : int
            Doctest option flags (e.g., ELLIPSIS, NORMALIZE_WHITESPACE).

        """
        self.doctest_parser = DocTestStringParser(DocTestEvaluator(doctest_optionflags))
        self.codeblock_parser = myst.CodeBlockParser(language="pycon")

    def __call__(self, document: Document) -> Iterable[Region]:
        """Parse pycon code blocks and yield doctest regions.

        Parameters
        ----------
        document : Document
            The Sybil document to parse.

        Yields
        ------
        Region
            Parsed regions from either pycon doctest blocks or regular code blocks.

        """
        for region in self.codeblock_parser(document):
            source = region.parsed
            # Treat pycon blocks as doctests (always start with >>>)
            if isinstance(source, str) and source.startswith(">>>"):
                for doctest_region in self.doctest_parser(source, document.path):
                    doctest_region.adjust(region, source)
                    yield doctest_region
            else:
                yield region


parsers: Sequence[Callable[[Document], Iterable[Region]]] = [
    myst.DocTestDirectiveParser(optionflags=optionflags),
    myst.PythonCodeBlockParser(doctest_optionflags=optionflags),
    PyconCodeBlockParser(doctest_optionflags=optionflags),
    myst.SkipParser(),
]

docs = Sybil(parsers=parsers, patterns=["*.md"])
python = Sybil(
    parsers=[*parsers, rest.DocTestParser(optionflags=optionflags), rest.SkipParser()],
    patterns=["*.py"],
)


pytest_collect_file = (docs + python).pytest()
