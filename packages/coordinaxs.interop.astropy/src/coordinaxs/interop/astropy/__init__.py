"""Coordinax interoperability with Astropy."""

__all__: tuple[str, ...] = (
    "convert_cx_cdict_to_astropy_cartrep",
    "convert_cx_cdict_to_astropy_cylrep",
    "convert_cx_cdict_to_astropy_physsphrep",
    "convert_cx_cdict_to_astropy_radialrep",
    "convert_cx_cdict_to_astropy_sphrep",
    "convert_cx_cdict_to_astropy_unitsphrep",
)

from ._setup_package import install_import_hook
from ._version import version as __version__  # noqa: F401

with install_import_hook("coordinaxs.interop.astropy"):
    from ._src import (
        convert_cx_cdict_to_astropy_cartrep,
        convert_cx_cdict_to_astropy_cylrep,
        convert_cx_cdict_to_astropy_physsphrep,
        convert_cx_cdict_to_astropy_radialrep,
        convert_cx_cdict_to_astropy_sphrep,
        convert_cx_cdict_to_astropy_unitsphrep,
    )

del install_import_hook
