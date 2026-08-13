"""``coordinax.internal`` — semi-public utilities.

.. warning::

    Everything in this module is **semi-public**.  The APIs exposed here
    are usable by downstream packages but are **not** covered by the
    same stability guarantees as the top-level ``coordinax`` API.  Names,
    signatures, and behaviour may change **at any time without warning**
    in minor or patch releases.  Pin to an exact version if you depend on
    anything here.

Contents:

- ``pack_uniform_unit``
    Pack dict-of-quantities into an array, converting all entries to
    a common unit.

- ``tree_cast_int_bool_to_float``
    Tree-map over a PyTree, promoting integer and boolean leaves to the
    default floating-point dtype (``jax.dtypes.canonicalize_dtype(jnp.float_)``).
    Existing float and complex leaves are left unchanged.  Useful for
    satisfying ``jax.jacfwd``'s requirement of real-floating inputs.

- ``Shape``, ``OptUSys``
    Shared type vocabulary, re-exported from ``coordinax._src.custom_types`` so
    downstream packages need not restate it.  ``CKey`` and ``CDict`` are public
    in `coordinaxs.api.custom_types`.

"""

__all__ = (
    "tree_cast_int_bool_to_float",
    "pack_uniform_unit",
    "pos_named_objs",
    "jax_scalar_handler",
    # Types
    "OptUSys",
    "Shape",
)

from ._src.setup_package import install_import_hook

with install_import_hook("coordinax.internal"):
    from coordinax._src.custom_types import OptUSys, Shape
    from coordinax._src.internal import (
        jax_scalar_handler,
        pack_uniform_unit,
        pos_named_objs,
        tree_cast_int_bool_to_float,
    )

del install_import_hook
