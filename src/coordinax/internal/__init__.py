"""`coordinax.internal` — semi-public utilities.

.. warning::

    Everything in this module is **semi-public**.  The APIs exposed here
    are usable by downstream packages but are **not** covered by the
    same stability guarantees as the top-level ``coordinax`` API.  Names,
    signatures, and behaviour may change **at any time without warning**
    in minor or patch releases.  Pin to an exact version if you depend on
    anything here.

Contents:

- ``QuantityMatrix``
    An N-D quantity matrix/vector where every element carries its own unit.
    Supports both 1-D (vector) and 2-D (matrix) cases.
    Useful for Jacobians and metric tensors whose entries have
    heterogeneous physical dimensions.

- ``UnitsMatrix``
    Nested tuple of units with indexing support for 1-D, 2-D (and N-D).

- ``pack_uniform_unit``
    Pack dict-of-quantities into an array, converting all entries to
    a common unit.

- ``structured``
    Decorator for transparent argument and return value processing.
    This helps pushing the logic for packing/unpacking inside a JIT.

"""

__all__ = (
    "QuantityMatrix",
    "UnitsMatrix",
    "pack_uniform_unit",
    "cdict_units",
    "pack_nonuniform_unit",
    "pack_with_usys",
    "pack_to_qmatrix",
)

from ._setup_package import install_import_hook

with install_import_hook("coordinax.internal"):
    from coordinax.internal._pack_utils import (
        pack_nonuniform_unit,
        pack_to_qmatrix,
        pack_uniform_unit,
        pack_with_usys,
    )
    from coordinax.internal._quantity_matrix import (
        QuantityMatrix,
        UnitsMatrix,
        cdict_units,
    )

del install_import_hook
