"""`coordinax.internal` — semi-public utilities.

.. warning::

    Everything in this module is **semi-public**.  The APIs exposed here
    are usable by downstream packages but are **not** covered by the
    same stability guarantees as the top-level ``coordinax`` API.  Names,
    signatures, and behaviour may change **at any time without warning**
    in minor or patch releases.  Pin to an exact version if you depend on
    anything here.

Contents:

- ``QMatrix``
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

from . import custom_types  # noqa: F401
from .dtype_utils import *
from .pack_utils import *
from .quantity_matrix import *
from .wl_utils import *
