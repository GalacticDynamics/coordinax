"""Coordinax interoperability with Astropy."""

__all__: tuple[str, ...] = (
    "convert_cx_cdict_to_astropy_cartrep",
    "convert_cx_cdict_to_astropy_cylrep",
    "convert_cx_cdict_to_astropy_physsphrep",
    "convert_cx_cdict_to_astropy_sphrep",
)

from ._src import (
    convert_cx_cdict_to_astropy_cartrep,
    convert_cx_cdict_to_astropy_cylrep,
    convert_cx_cdict_to_astropy_physsphrep,
    convert_cx_cdict_to_astropy_sphrep,
)
