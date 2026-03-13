"""`coordinax.vectors` Module."""

__all__ = (
    "AbstractVector",
    "Vector",
    "vconvert",
    "ToUnitsOptions",
)

from ._setup_package import install_import_hook

with install_import_hook("coordinax.vectors"):
    from ._src import (
        AbstractVector,
        ToUnitsOptions,
        Vector,
    )
    from coordinax.api.representations import vconvert

del install_import_hook
