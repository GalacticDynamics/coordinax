"""Shared internal helpers for the manifolds subpackage."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array

import unxt as u
import unxts.linalg as ul

DMLS = u.unit("")


def as_quantity_matrix(x: ul.QM | Array, /) -> ul.QM:
    """Return *x* as a `QM`, wrapping a plain array as a dimensionless matrix.

    `ul.QM` is an alias of `ul.QuantityMatrix`, so a unitful matrix of either
    spelling passes through untouched and keeps its units.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxts.linalg as ul
    >>> from coordinax._src.manifolds._utils import as_quantity_matrix

    >>> as_quantity_matrix(jnp.eye(2))
    QM([[1., 0.],
        [0., 1.]], '((, ), (, ))')

    >>> g = ul.QuantityMatrix(jnp.eye(2), unit=ul.UnitsMatrix.full((2, 2), "m"))
    >>> as_quantity_matrix(g) is g
    True

    """
    if isinstance(x, ul.QM):
        return x
    n_rows, n_cols = x.shape[-2:]
    return ul.QM(value=x, unit=ul.UnitsMatrix.full((n_rows, n_cols), DMLS))
