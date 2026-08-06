"""Optional dependencies. Internal use only."""

__all__ = ("OptDeps",)

from optional_dependencies import OptionalDependencyEnum, auto


class OptDeps(OptionalDependencyEnum):  # type: ignore[misc]  # pylint: disable=invalid-enum-extension
    """Optional dependencies for ``coordinaxs.astro``.

    Member names are canonicalized to distribution names, so
    ``UNXTS_PARAMETRIC`` resolves ``unxts.parametric``.

    Keep at most one ``unxts.*`` member here. :class:`OptionalDependencyEnum`
    keys each member on its installed *version*, and equal values make an
    `enum.Enum` collapse the later member into an alias of the earlier one --
    silently, with no error. The ``unxts.*`` sub-packages are released together
    and so usually share a version (``unxts.api``, ``unxts.hypothesis`` and
    ``unxts.parametric`` are all 2.0.0 as of writing), which makes version an
    unsafe key for telling them apart. A second one belongs behind an
    import-spec lookup instead -- `optional_dependencies.utils.is_installed`
    resolves by module name, independent of version.
    """

    UNXTS_PARAMETRIC = auto()
