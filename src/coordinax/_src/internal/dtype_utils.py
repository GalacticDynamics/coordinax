"""Internal dtype helpers.

Utilities in this module provide small, reusable dtype-normalization helpers
for JAX arrays and PyTree structures used across coordinax internals.
"""

__all__: tuple[str, ...] = ("tree_cast_int_bool_to_float",)

from jaxtyping import Array, Bool, Complex, Float, Int, PyTree
from typing import Final

import jax
import jax.numpy as jnp

DEFAULT_FLOAT_DTYPE: Final = jax.dtypes.canonicalize_dtype(jnp.float_)

NumericLeaf = (
    Bool[Array, "..."] | Int[Array, "..."] | Float[Array, "..."] | Complex[Array, "..."]
)
InexactLeaf = Float[Array, "..."] | Complex[Array, "..."]


def _cast_int_bool_leaf_to_float(x: NumericLeaf, /) -> InexactLeaf:
    """Cast integer/bool leaves to the configured default floating dtype."""
    dtype = x.dtype
    if jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(dtype, jnp.bool_):
        return x.astype(DEFAULT_FLOAT_DTYPE)
    return x


def tree_cast_int_bool_to_float(tree: PyTree[NumericLeaf], /) -> PyTree[InexactLeaf]:
    """Tree-map integer/bool leaves to the configured default float dtype.

    This intentionally does not cast complex leaves, which prevents silent
    imaginary-part loss.

    >>> import jax.numpy as jnp
    >>> from coordinax.internal import tree_cast_int_bool_to_float

    >>> x = {
    ...     "i": jnp.array([1, 2], dtype=jnp.int32),
    ...     "b": jnp.array([True, False], dtype=jnp.bool_),
    ...     "f": jnp.array([1.5], dtype=jnp.float32),
    ...     "c": jnp.array([1 + 2j], dtype=jnp.complex64),
    ... }
    >>> tree_cast_int_bool_to_float(x)
    {'b': Array([1., 0.], dtype=float64),
     'c': Array([1.+2.j], dtype=complex64),
     'f': Array([1.5], dtype=float32),
     'i': Array([1., 2.], dtype=float64)}

    """
    return jax.tree.map(_cast_int_bool_leaf_to_float, tree)
