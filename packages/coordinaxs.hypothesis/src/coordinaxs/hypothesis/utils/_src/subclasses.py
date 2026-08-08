"""Utilities."""

__all__ = ("get_all_subclasses",)

import functools as ft
import importlib
import inspect
import sys
import warnings

from typing import Final


@ft.cache
def _public_coordinax_module_candidates(module: str, /) -> tuple[str, ...]:
    """Build candidate public module names for a coordinax class module."""
    public_module = module.replace("._src.", ".")

    candidates: list[str] = [public_module]

    # Walk up module parents (e.g. coordinax.charts.foo -> coordinax.charts).
    current = public_module
    while "." in current:
        current = current.rsplit(".", 1)[0]
        candidates.append(current)
        if current == "coordinax":
            break

    # Include already-loaded public coordinax modules for re-exports.
    loaded_public = sorted(
        name
        for name in sys.modules
        if name.startswith("coordinax.") and "._src." not in name
    )
    candidates.extend(loaded_public)

    return tuple(dict.fromkeys(candidates))


@ft.cache
def canonicalize_coordinax_class(cls: type, /) -> type:
    """Resolve a coordinax class to its canonical version.

    In editable installs with uv-workspaces, the same class can exist as
    multiple Python objects due to import path duplication. This function
    returns the canonical version by looking it up via its ``__qualname__``
    across likely public coordinax modules.

    Parameters
    ----------
    cls : type
        The class to canonicalize.

    Returns
    -------
    type
        The canonical version of the class from public coordinax modules, or
        the original class if it can't be resolved.

    """
    module = getattr(cls, "__module__", "")

    # Only process coordinax classes
    if not module.startswith("coordinax"):
        return cls

    parts = tuple(cls.__qualname__.split("."))

    for mod_name in _public_coordinax_module_candidates(module):
        mod = sys.modules.get(mod_name)
        if mod is None:
            try:
                mod = importlib.import_module(mod_name)
            except ImportError:  # pragma: no cover - optional namespace availability
                continue

        try:
            resolved: object = mod
            for part in parts:
                resolved = getattr(resolved, part)
            if isinstance(resolved, type):
                return resolved
        except AttributeError:
            continue

    return cls


#: Module-path components that mark a class as declared by a test rather than by
#: a library: a ``tests`` package, a ``test_*`` or ``conftest`` module, or an
#: interactive session.
_TEST_MODULE_PARTS: Final = frozenset({"tests", "test", "conftest", "__main__"})


def is_test_declared(cls: type, /) -> bool:
    """Whether *cls* was declared by a test rather than by a library.

    The subclass tree is process-global, so every fake a test declares sits in
    it beside the real types and gets drawn as though it were one. Lifetime
    cannot separate them -- ``__subclasses__`` is already weak, and a fake stays
    alive as long as the module that declared it is imported -- so this goes on
    where the class was defined.

    Deliberately keyed on *test* provenance rather than on "outside coordinax":
    a downstream library that defines its own charts is a first-class user of
    these strategies and its types must still be drawn. Only classes from test
    modules are dropped, which also covers that library's own fakes.
    """
    module = getattr(cls, "__module__", "")
    parts = module.split(".")
    return bool(_TEST_MODULE_PARTS.intersection(parts)) or parts[-1].startswith("test_")


def is_library_class(cls: type, /) -> bool:
    """Whether *cls* belongs to coordinax itself.

    Used to decide whether `get_all_subclasses` is being asked about coordinax's
    own hierarchy, in which case it drops test-declared subclasses, or about a
    caller's, in which case it stays a plain subclass walk. The ``coordinax``
    prefix covers the plugin distributions too, matching
    `canonicalize_coordinax_class`.
    """
    return getattr(cls, "__module__", "").startswith("coordinax")


def is_abstract_class(cls: type, /) -> bool:
    """Determine if a class is abstract."""
    return inspect.isabstract(cls) or cls.__name__.startswith("Abstract")


@ft.cache
def get_all_subclasses(
    base_class: type,
    /,
    *,
    filter: type | tuple[type, ...] = object,
    exclude_abstract: bool = True,
    exclude: tuple[type, ...] = (),
) -> tuple[type, ...]:
    """Build a set of all subclasses of a given base class.

    Recursively walks the subclass tree of *base_class*, deduplicating and
    (optionally) filtering the results in {class}`coordinax`.  The return value
    is cached via {func}`functools.cache`.

    When *base_class* belongs to coordinax, subclasses declared by test modules
    are skipped -- see `is_test_declared`. The subclass tree is process-global,
    so without that every fake a test declares would be handed out as though it
    were a real type, and *whether* it was would depend on when the cache
    happened to warm.

    Classes from ordinary library modules are always kept, **including a
    downstream package's own** -- extending coordinax and having your types
    drawn is a supported use of these strategies. And for a base of your own the
    walk stays entirely plain.

    .. note::

        The cache is keyed on the arguments while ``__subclasses__()`` is
        mutable, so a coordinax class imported after the first call stays absent
        until `cache_clear`. That window is harmless in practice -- the plugin
        distributions register their types at import, before any strategy runs.

    Parameters
    ----------
    base_class : type
        The base class to find subclasses of.
    filter : type | tuple[type, ...], optional
        One or more classes that every returned subclass must also be a subclass
        of (AND semantics).  By default ``object``, which accepts everything.
    exclude_abstract : bool, optional
        Whether to exclude abstract subclasses, by default ``True``.  A class is
        considered abstract if it satisfies {func}`inspect.isabstract` **or**
        its name starts with ``"Abstract"``.
    exclude : tuple[type, ...], optional
        Specific classes (covariant) to exclude — any subclass of an excluded
        class is also excluded.  By default ``()``.

    Returns
    -------
    tuple[type, ...]
        A tuple of all matching subclasses of *base_class*.

    Warns
    -----
    UserWarning
        If no subclasses are found after filtering.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> from coordinaxs.hypothesis.utils import get_all_subclasses

    Find all concrete chart classes:

    >>> get_all_subclasses.cache_clear()
    >>> result = get_all_subclasses(cxc.AbstractChart)
    >>> isinstance(result, tuple)
    True
    >>> cxc.Cart3D in result
    True
    >>> cxc.Spherical3D in result
    True

    Exclude a specific class (and its subclasses):

    >>> get_all_subclasses.cache_clear()
    >>> result = get_all_subclasses(cxc.AbstractChart, exclude=(cxc.Cart3D,))
    >>> cxc.Cart3D in result
    False

    Include abstract classes by setting ``exclude_abstract=False``:

    >>> get_all_subclasses.cache_clear()
    >>> concrete = get_all_subclasses(cxc.AbstractChart, exclude_abstract=True)
    >>> get_all_subclasses.cache_clear()
    >>> with_abstract = get_all_subclasses(
    ...     cxc.AbstractChart, exclude_abstract=False
    ... )
    >>> len(with_abstract) > len(concrete)
    True

    """
    # Use a dict keyed by (module, qualname) to deduplicate classes that appear
    # multiple times due to import path issues in editable installs.
    seen: dict[tuple[str, str], type] = {}

    # Normalize filter to a tuple
    filter_tuple = filter if isinstance(filter, tuple) else (filter,)
    canonical_filter = tuple(canonicalize_coordinax_class(cls) for cls in filter_tuple)
    canonical_exclude = tuple(canonicalize_coordinax_class(cls) for cls in exclude)

    # Only police coordinax's own hierarchies. Asked about a base of your own,
    # this stays a plain subclass walk -- which is what the utility's other
    # callers, and their synthetic test hierarchies, rely on.
    drop_test_classes = is_library_class(base_class)

    def recurse(cls: type, /) -> None:
        for subclass in cls.__subclasses__():
            # Canonicalize early so checks below are robust to duplicate module
            # identities in editable/workspace installs.
            canonical = canonicalize_coordinax_class(subclass)

            # Skip if in exclude list
            if any(issubclass(canonical, ex) for ex in canonical_exclude):
                continue

            # Check if subclass matches ALL filters (not just ANY)
            if (
                not (drop_test_classes and is_test_declared(canonical))
                and all(issubclass(canonical, f) for f in canonical_filter)
                and not (exclude_abstract and is_abstract_class(canonical))
            ):
                # Deduplicate by (module, qualname) - only keep first seen
                key = (canonical.__module__, canonical.__qualname__)
                if key not in seen:
                    seen[key] = canonical

            # Always recurse to find deeper subclasses
            recurse(subclass)

    recurse(base_class)

    subclasses = list(seen.values())

    if not subclasses:
        warnings.warn(
            f"No subclasses found for base class {base_class} "
            f"with filter={filter} "
            f"and exclude_abstract={exclude_abstract}.",
            category=UserWarning,
            stacklevel=2,
        )

    return tuple(subclasses)
