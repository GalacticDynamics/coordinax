"""coordinax: Coordinates in JAX."""
# pylint: disable=import-error

__all__ = (
    # modules
    "angles",
    "charts",
    "distances",
    "embeddings",
    "frames",
    "metrics",
    "objs",
    "ops",
    "roles",
    "transforms",
    # specialized Quantity objects
    "Angle",
    "Distance",
    # High-level objects
    "as_pos",
    "vconvert",
    "Vector",
    "PointedVector",
    "Coordinate",
    "cdict",
    # Convenience access to charts
    "cart3d",
)

from .setup_package import install_import_hook

with install_import_hook("coordinax"):
    from . import (
        angles,
        charts,
        distances,
        embeddings,
        frames,
        metrics,
        objs,
        ops,
        roles,
        transforms,
    )
    from ._version import version as __version__  # noqa: F401
    from .angles import Angle
    from .charts import cart3d
    from .distances import Distance
    from .objs import Coordinate, PointedVector, Vector, as_pos, cdict, vconvert

# isort: split
# Interoperability - import the module but don't trigger registration yet
from . import _interop

# Now that coordinax is fully loaded, register interop packages
# This avoids circular import issues since interop packages depend on coordinax
_interop._register_interop_packages()

# Cleanup
del _interop, install_import_hook
