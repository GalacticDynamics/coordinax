"""Utilities."""

__all__ = ("get_all_subclasses",)

import functools as ft
import inspect
import sys
import warnings

from typing import Any

import coordinax.charts as cxc


def canonicalize_coordinax_class(cls: type, /) -> type:
    """Resolve a coordinax class to its canonical version.

    In editable installs with uv-workspaces, the same class can exist as
    multiple Python objects due to import path duplication. This function
    returns the canonical version by looking it up via its __qualname__ in
    the coordinax.charts module.

    Parameters
    ----------
    cls : type
        The class to canonicalize.

    Returns
    -------
    type
        The canonical version of the class from coordinax.charts, or the
        original class if it can't be resolved.

    """
    module = getattr(cls, "__module__", "")

    # Only process coordinax classes
    if not module.startswith("coordinax"):
        return cls

    # Try to resolve via coordinax.charts using __qualname__
    parts = cls.__qualname__.split(".")
    try:
        resolved: Any = cxc
        for part in parts:
            resolved = getattr(resolved, part)
        if isinstance(resolved, type):
            return resolved
    except AttributeError:
        pass

    # Fallback: try the public module path from sys.modules
    public_module = module.replace("._src.", ".")
    if public_module in sys.modules:
        try:
            resolved = sys.modules[public_module]
            for part in parts:
                resolved = getattr(resolved, part)
            if isinstance(resolved, type):
                return resolved
        except AttributeError:
            pass

    return cls


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
    >>> from coordinax_hypothesis.utils import get_all_subclasses

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

    def recurse(cls: type, /) -> None:
        for subclass in cls.__subclasses__():
            # Skip if in exclude list
            if any(issubclass(subclass, ex) for ex in exclude):
                continue
            # Check if subclass matches ALL filters (not just ANY)
            if all(issubclass(subclass, f) for f in filter_tuple) and not (
                exclude_abstract and is_abstract_class(subclass)
            ):
                # Canonicalize the class to handle duplicates from editable
                # installs in uv-workspaces.
                canonical = canonicalize_coordinax_class(subclass)

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
