"""Optional dependencies. Internal use only."""

__all__ = ["OptDeps"]

from optional_dependencies import OptionalDependencyEnum, auto


class OptDeps(OptionalDependencyEnum):  # type: ignore[misc]  # pylint: disable=invalid-enum-extension
    """Optional dependencies for ``coordinax``."""

    ASTROPY = auto()
