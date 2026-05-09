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

- ``tree_cast_int_bool_to_float``
    Tree-map over a PyTree, promoting integer and boolean leaves to the
    default floating-point dtype (``jax.dtypes.canonicalize_dtype(jnp.float_)``).
    Existing float and complex leaves are left unchanged.  Useful for
    satisfying ``jax.jacfwd``'s requirement of real-floating inputs.

- ``structured``
    Decorator for transparent argument and return value processing.
    This helps pushing the logic for packing/unpacking inside a JIT.

"""

__all__ = (
    "QuantityMatrix",
    "UnitsMatrix",
    "tree_cast_int_bool_to_float",
    "pack_uniform_unit",
    "cdict_units",
    "pack_nonuniform_unit",
    "pack_with_usys",
    "pack_to_qmatrix",
    "pos_named_objs",
    "jax_scalar_handler",
)

from ._src.setup_package import install_import_hook

with install_import_hook("coordinax.internal"):
    from coordinax._src.internal import (
        QuantityMatrix,
        UnitsMatrix,
        cdict_units,
        jax_scalar_handler,
        pack_nonuniform_unit,
        pack_to_qmatrix,
        pack_uniform_unit,
        pack_with_usys,
        pos_named_objs,
        tree_cast_int_bool_to_float,
    )

del install_import_hook
