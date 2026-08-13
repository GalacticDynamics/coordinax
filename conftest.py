"""Doctest configuration."""

import contextlib
import importlib
import os
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

# Deadlines measure wall-clock time per example, and JAX compiles on an
# example's first execution: the same example takes ~400ms then and ~2ms on the
# replay hypothesis does to confirm it. That is reported as `DeadlineExceeded`
# or `FlakyFailure` -- a real failure, at a random example, on a loaded machine.
# It is off everywhere rather than per test, which is what the 100+ scattered
# `@settings(deadline=None)` decorators were doing one at a time.
DEADLINE = None

# Quick smoke test profile to check the test infrastructure is working.
settings.register_profile(
    "smoke",
    deadline=DEADLINE,
    max_examples=5,
    phases=[Phase.explicit, Phase.reuse, Phase.generate],
    suppress_health_check=[HealthCheck.too_slow],
)

# Default profile for development: more examples and allow slow tests.
settings.register_profile(
    "dev",
    deadline=DEADLINE,
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
)

# Thorough profile for CI: many examples and all health checks.
settings.register_profile("thorough", deadline=DEADLINE, max_examples=500)

# What runs unless `HYPOTHESIS_PROFILE` names another. Hypothesis's own
# defaults otherwise, so only the deadline changes. Without this `load_profile`
# none of the profiles above was ever selected -- `thorough`'s 500 examples had
# never run anywhere.
settings.register_profile("default", deadline=DEADLINE)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


# =========================================================
# Paths and Namespaces

CX_WORKSPACE_ROOT = pathlib.Path(__file__).parent
CX_PACKAGES_ROOT = CX_WORKSPACE_ROOT / "packages"

#: ``(src directory, namespace package name)`` for every workspace source tree.
#:
#: The core distribution ships ``src/coordinax``; every workspace sub-package
#: ships ``src/coordinaxs``, the shared PEP 420 namespace.
SrcRoot = tuple[pathlib.Path, str]


def _discover_src_roots() -> tuple[SrcRoot, ...]:
    """Every workspace ``src`` directory paired with its namespace name."""
    candidates: list[SrcRoot] = [(CX_WORKSPACE_ROOT / "src", "coordinax")]
    if CX_PACKAGES_ROOT.exists():
        candidates += [
            (package_dir / "src", "coordinaxs")
            for package_dir in sorted(CX_PACKAGES_ROOT.iterdir())
        ]
    # Order-preserving dedup. In a normal checkout it removes nothing -- each
    # sub-package has its own `src`, so the pairs already differ despite
    # sharing the `coordinaxs` namespace name. It matters only when two
    # candidates `.resolve()` to the same directory, e.g. a symlinked package
    # dir. The hand-rolled `unique_roots` loop this replaces guarded the same
    # case.
    return tuple(
        dict.fromkeys(
            (src.resolve(), namespace)
            for src, namespace in candidates
            if (src / namespace).exists()
        )
    )


SRC_ROOTS: tuple[SrcRoot, ...] = _discover_src_roots()

#: ``(src root, namespace name)`` -- for mapping a file to its import root.
RESOLVED_PACKAGE_ROOTS: tuple[SrcRoot, ...] = SRC_ROOTS

#: ``(namespace directory, namespace parts)`` -- for mapping a file to a module.
RESOLVED_MODULE_ROOTS: tuple[tuple[pathlib.Path, tuple[str, ...]], ...] = tuple(
    (src / namespace, (namespace,)) for src, namespace in SRC_ROOTS
)


_ORIG_RESOLVE_PACKAGE_PATH = pytest_pathlib.resolve_package_path
_ORIG_RESOLVE_PKG_ROOT_AND_MODULE = pytest_pathlib.resolve_pkg_root_and_module_name


def _resolve_package_path_with_namespace(path: pathlib.Path) -> pathlib.Path | None:
    """Resolve package path with PEP 420 workspace roots as namespace packages.

    This ensures files under ``*/src/coordinax`` and ``*/src/coordinaxs`` are
    collected/imported with the canonical ``coordinax.*`` / ``coordinaxs.*``
    module path instead of short aliases such as ``charts.*``.
    """
    resolved = path.resolve()
    for root, namespace_dir in RESOLVED_PACKAGE_ROOTS:
        namespace_root = root / namespace_dir
        if resolved.is_relative_to(namespace_root):
            # Match the wrapped function's contract: return the top-level
            # package directory, not its parent. Callers derive the import root
            # as ``resolve_package_path(path).parent``, so returning ``root``
            # here would yield module names like ``src.coordinax.charts``.
            return namespace_root
    return _ORIG_RESOLVE_PACKAGE_PATH(path)


pytest_pathlib.resolve_package_path = _resolve_package_path_with_namespace  # ty: ignore[invalid-assignment]


def _resolve_pkg_root_and_module_name_with_namespace(
    path: pathlib.Path, *, consider_namespace_packages: bool = False
) -> tuple[pathlib.Path, str]:
    """Resolve canonical package root and module name for workspace namespace files."""
    resolved = path.resolve()
    for root, namespace_dir in RESOLVED_PACKAGE_ROOTS:
        namespace_root = root / namespace_dir
        if resolved.is_relative_to(namespace_root):
            module_name = pytest_pathlib.compute_module_name(root, resolved)
            if module_name:
                return root, module_name

    return _ORIG_RESOLVE_PKG_ROOT_AND_MODULE(
        path, consider_namespace_packages=consider_namespace_packages
    )


pytest_pathlib.resolve_pkg_root_and_module_name = (  # ty: ignore[invalid-assignment]
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
    module identities versus ``coordinax._src.charts``.
    """
    resolved_path = path.resolve()
    for root, namespace in RESOLVED_MODULE_ROOTS:
        if resolved_path.is_relative_to(root):
            return importlib.import_module(
                _path_to_module(resolved_path, root, namespace)
            )

    return sybil_import_path(path)


sybil_document.import_path = _import_path_with_namespace  # ty: ignore[invalid-assignment]
sybil_python.import_path = _import_path_with_namespace  # ty: ignore[invalid-assignment]


# =========================================================
# Canonical coordinax namespace preloading


def _preload_coordinax_namespace() -> None:
    """Preload key coordinax modules via canonical names after hook install.

    This must run only after pytest and Sybil import-path hooks above are
    installed; otherwise the same logical modules may be loaded under multiple
    non-canonical names.
    """
    # `coordinax` is a hard requirement of the suite, so a failure to import it
    # is a real error rather than an absent optional package and must not be
    # suppressed — swallowing it here would surface much later as confusing
    # collection failures. (Import order is not otherwise load-bearing: interop
    # registration is order-independent, see `coordinax._load_optional_interop`.)
    importlib.import_module("coordinax")

    # Optional/auxiliary modules: absent in minimal installs.
    module_names = (
        "coordinaxs.api",
        "coordinaxs.api.charts",
        "coordinaxs.api.frames",
        "coordinaxs.api.manifolds",
        "coordinaxs.api.representations",
        "coordinaxs.astro",
        "coordinax.charts",
        "coordinax.frames",
        "coordinaxs.hypothesis",
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


class MystCodeBlockParser:
    """Parse a MyST code block, routing ``>>>`` content to the doctest evaluator.

    Covers both block flavours the docs use. They differ only in the language
    tag, whether a `PythonEvaluator` runs the non-doctest case, and whether the
    source needs cleaning first:

    * ``pycon``          -- doctest or plain region, no evaluator.
    * ``{code-cell} ipython3`` -- run as Python, with IPython line-magics
      (``%...``) stripped so cells authored for Jupyter still compile.
    """

    def __init__(
        self,
        language: str,
        doctest_optionflags: int = 0,
        *,
        evaluator: object | None = None,
        transform: Callable[[Lexeme], Lexeme] | None = None,
    ) -> None:
        """Build a parser for *language*, optionally evaluating and cleaning."""
        self.doctest_parser = DocTestStringParser(DocTestEvaluator(doctest_optionflags))
        kwargs = {} if evaluator is None else {"evaluator": evaluator}
        self.codeblock_parser = myst.CodeBlockParser(language=language, **kwargs)
        self.transform = transform

    def __call__(self, document: Document) -> Iterable[Region]:
        """Yield doctest regions for ``>>>`` blocks, code regions otherwise."""
        for region in self.codeblock_parser(document):
            source = region.parsed
            if isinstance(source, str) and source.startswith(">>>"):
                for doctest_region in self.doctest_parser(source, document.path):
                    doctest_region.adjust(region, source)
                    yield doctest_region
            else:
                if self.transform is not None:
                    region.parsed = self.transform(source)
                yield region


def _strip_ipython_magics(source: Lexeme) -> Lexeme:
    """Remove IPython line-magic lines (``%...``) from *source*."""
    cleaned = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("%")
    )
    return Lexeme(cleaned, source.offset, source.line_offset)


optionflags = ELLIPSIS | NORMALIZE_WHITESPACE

parsers: Sequence[Callable[[Document], Iterable[Region]]] = [
    myst.DocTestDirectiveParser(optionflags=optionflags),
    myst.PythonCodeBlockParser(doctest_optionflags=optionflags),
    MystCodeBlockParser("pycon", doctest_optionflags=optionflags),
    MystCodeBlockParser(
        "ipython3",
        doctest_optionflags=optionflags,
        evaluator=PythonEvaluator(),
        transform=_strip_ipython_magics,
    ),
    myst.SkipParser(),
]

docs = Sybil(parsers=parsers, patterns=["*.md"])
python = Sybil(
    parsers=[*parsers, rest.DocTestParser(optionflags=optionflags), rest.SkipParser()],
    patterns=["*.py"],
)


pytest_collect_file = (docs + python).pytest()
