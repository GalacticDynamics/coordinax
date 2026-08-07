"""Optional dependencies. Internal use only."""

__all__ = ("OptDeps",)

from optional_dependencies import OptionalDependencyEnum, auto


class OptDeps(OptionalDependencyEnum):  # pylint: disable=invalid-enum-extension
    """Optional dependencies for ``coordinax``.

    Member names are canonicalized to distribution names, so
    ``UNXTS_PARAMETRIC`` resolves ``unxts.parametric``. Add members freely:
    they stay distinct even when they share a version.
    """

    UNXTS_PARAMETRIC = auto()
