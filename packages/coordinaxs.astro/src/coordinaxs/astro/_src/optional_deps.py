"""Optional dependencies. Internal use only."""

__all__ = ("OptDeps",)

from optional_dependencies import OptionalDependencyEnum, auto


class OptDeps(OptionalDependencyEnum):  # type: ignore[misc]  # pylint: disable=invalid-enum-extension
    """Optional dependencies for ``coordinaxs.astro``.

    Member names are canonicalized to distribution names, so
    ``UNXTS_PARAMETRIC`` resolves ``unxts.parametric``.

    Add further members freely, including other ``unxts.*`` sub-packages. Up to
    ``optional-dependencies`` 0.4.0 that was unsafe: members were keyed on the
    installed *version*, so any two sharing one -- every pair of uninstalled
    dependencies, and the co-released ``unxts.*`` packages, which usually share
    a version -- silently collapsed into a single `enum.Enum` member reporting
    the wrong package's state. 0.5.0 keys members so they stay distinct, hence
    the floor in ``pyproject.toml``.
    """

    UNXTS_PARAMETRIC = auto()
