"""Shared internal helpers for the manifolds subpackage."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array

import unxts.linalg as ul

import unxt as u

DMLS = u.unit("")


def as_quantity_matrix(x: ul.QM | Array, /) -> ul.QM:
    """Return *x* as a `QM`, wrapping a plain array as a dimensionless matrix."""
    if isinstance(x, ul.QM):
        return x
    n_rows, n_cols = x.shape[-2:]
    return ul.QM(value=x, unit=ul.UnitsMatrix.full((n_rows, n_cols), DMLS))
