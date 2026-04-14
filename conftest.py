"""Doctest configuration."""

import contextlib
import importlib
import pathlib
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

from collections.abc import Callable, Iterable, Sequence
from types import ModuleType

import _pytest.pathlib as pytest_pathlib
import sybil.document as sybil_document
import sybil.python as sybil_python
from hypothesis import HealthCheck, Phase, settings
from sybil import Document, Lexeme, Region, Sybil, document as sybil_document
from sybil.evaluators.doctest import DocTestEvaluator
from sybil.evaluators.python import PythonEvaluator
from sybil.parsers import myst, rest
from sybil.parsers.abstract.doctest import DocTestStringParser
from sybil.python import import_path as sybil_import_path

# =========================================================
# Hypothesis settings

# Quick smoke test profile to check the test infrastructure is working.
settings.register_profile(
    "smoke", max_examples=5, phases=[Phase.explicit, Phase.reuse, Phase.generate]
)

# Default profile for development: more examples and allow slow tests.
settings.register_profile(
    "dev", max_examples=50, suppress_health_check=[HealthCheck.too_slow]
)

# Thorough profile for CI: many examples and all health checks.
settings.register_profile("thorough", max_examples=500)


# =========================================================
# Paths and Namespaces

ModuleRoot = tuple[pathlib.Path, tuple[str, ...]]
COORDINAX_NAMESPACE: tuple[str, ...] = ("coordinax",)

CX_WORKSPACE_ROOT = pathlib.Path(__file__).parent
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


def _discover_package_roots() -> tuple[pathlib.Path, ...]:
    """Discover all workspace ``src`` roots containing the ``coordinax`` namespace."""
    roots: list[pathlib.Path] = []

    main_src_root = CX_WORKSPACE_ROOT / "src"
    if (main_src_root / "coordinax").exists():
        roots.append(main_src_root.resolve())

    if CX_PACKAGES_ROOT.exists():
        for package_dir in sorted(CX_PACKAGES_ROOT.iterdir()):
            src_root = package_dir / "src"
            if (src_root / "coordinax").exists():
                roots.append(src_root.resolve())

    # Preserve order but deduplicate.
    unique_roots: list[pathlib.Path] = []
    for root in roots:
        if root not in unique_roots:
            unique_roots.append(root)
    return tuple(unique_roots)


RESOLVED_PACKAGE_ROOTS: tuple[pathlib.Path, ...] = _discover_package_roots()


_ORIG_RESOLVE_PACKAGE_PATH = pytest_pathlib.resolve_package_path
_ORIG_RESOLVE_PKG_ROOT_AND_MODULE = pytest_pathlib.resolve_pkg_root_and_module_name


def _resolve_package_path_with_namespace(path: pathlib.Path) -> pathlib.Path | None:
    """Resolve package path with PEP 420 workspace roots as namespace packages.

    This ensures files under ``*/src/coordinax`` are collected/imported with the
    canonical ``coordinax.*`` module path instead of short aliases such as
    ``charts.*``.
    """
    resolved = path.resolve()
    for root in RESOLVED_PACKAGE_ROOTS:
        if resolved.is_relative_to(root / "coordinax"):
            return root
    return _ORIG_RESOLVE_PACKAGE_PATH(path)


pytest_pathlib.resolve_package_path = _resolve_package_path_with_namespace  # type: ignore[assignment]


def _resolve_pkg_root_and_module_name_with_namespace(
    path: pathlib.Path, *, consider_namespace_packages: bool = False
) -> tuple[pathlib.Path, str]:
    """Resolve canonical package root and module name for workspace namespace files."""
    resolved = path.resolve()
    for root in RESOLVED_PACKAGE_ROOTS:
        namespace_root = root / "coordinax"
        if resolved.is_relative_to(namespace_root):
            module_name = pytest_pathlib.compute_module_name(root, resolved)
            if module_name:
                return root, module_name

    return _ORIG_RESOLVE_PKG_ROOT_AND_MODULE(
        path, consider_namespace_packages=consider_namespace_packages
    )


pytest_pathlib.resolve_pkg_root_and_module_name = (  # type: ignore[assignment]
    _resolve_pkg_root_and_module_name_with_namespace
)


# =========================================================
# Canonical import-path mapping for Sybil


def _path_to_module(
    path: pathlib.Path, root: pathlib.Path, namespace: tuple[str, ...], /
) -> str:
    """Map a Python file under ``root`` to a fully-qualified module path."""
    relative = path.relative_to(root)
    if relative.name == "__init__.py":
        suffix = tuple(relative.parts[:-1])
    else:
        suffix = (*tuple(relative.parts[:-1]), relative.stem)
    return ".".join((*namespace, *suffix))


def _import_path_with_namespace(path: pathlib.Path) -> ModuleType:
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


sybil_document.import_path = _import_path_with_namespace  # type: ignore[attr-defined]
sybil_python.import_path = _import_path_with_namespace  # ty: ignore[invalid-assignment]


# =========================================================
# Canonical coordinax namespace preloading


def _preload_coordinax_namespace() -> None:
    """Preload key coordinax modules via canonical names after hook install.

    This must run only after pytest and Sybil import-path hooks above are
    installed; otherwise the same logical modules may be loaded under multiple
    non-canonical names.
    """
    module_names = (
        "coordinax.api",
        "coordinax.api.charts",
        "coordinax.api.frames",
        "coordinax.api.manifolds",
        "coordinax.api.representations",
        "coordinax.astro",
        "coordinax.charts",
        "coordinax.frames",
        "coordinax.hypothesis",
        "coordinax.main",
        "coordinax.manifolds",
        "coordinax.representations",
        "coordinax.vectors",
    )
    for module_name in module_names:
        with contextlib.suppress(ModuleNotFoundError):
            importlib.import_module(module_name)


_preload_coordinax_namespace()


# =========================================================
# Sybil parser setup


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


class CodeCellParser:
    """Parser for MyST ``{code-cell} ipython3`` blocks.

    Treats the content as Python code (via ``PythonEvaluator``) or as a
    doctest when it starts with ``>>>``.  IPython line-magics (``%…``) are
    stripped before evaluation so that cells authored for Jupyter still
    compile as plain Python.
    """

    def __init__(self, doctest_optionflags: int = 0) -> None:  # noqa: D107
        self.doctest_parser = DocTestStringParser(DocTestEvaluator(doctest_optionflags))
        self.codeblock_parser = myst.CodeBlockParser(
            language="ipython3",
            evaluator=PythonEvaluator(),
        )

    @staticmethod
    def _strip_magics(source: Lexeme) -> Lexeme:
        """Remove IPython line-magic lines (``%…``) from *source*."""
        cleaned = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("%")
        )
        return Lexeme(cleaned, source.offset, source.line_offset)

    def __call__(self, document: Document) -> Iterable[Region]:  # noqa: D102
        for region in self.codeblock_parser(document):
            source = region.parsed
            if isinstance(source, str) and source.startswith(">>>"):
                for doctest_region in self.doctest_parser(source, document.path):
                    doctest_region.adjust(region, source)
                    yield doctest_region
            else:
                region.parsed = self._strip_magics(source)
                yield region


optionflags = ELLIPSIS | NORMALIZE_WHITESPACE

parsers: Sequence[Callable[[Document], Iterable[Region]]] = [
    myst.DocTestDirectiveParser(optionflags=optionflags),
    myst.PythonCodeBlockParser(doctest_optionflags=optionflags),
    PyconCodeBlockParser(doctest_optionflags=optionflags),
    CodeCellParser(doctest_optionflags=optionflags),
    myst.SkipParser(),
]

docs = Sybil(parsers=parsers, patterns=["*.md"])
python = Sybil(
    parsers=[*parsers, rest.DocTestParser(optionflags=optionflags), rest.SkipParser()],
    patterns=["*.py"],
)


pytest_collect_file = (docs + python).pytest()
