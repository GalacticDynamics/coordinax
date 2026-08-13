"""Shared internal helpers for the manifolds subpackage."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array
from typing import Any

import jax.numpy as jnp

import unxt as u
import unxts.linalg as ul
from unxt.quantity import is_any_quantity

from coordinax._src.base import AbstractMetricField

DMLS = u.unit("")


def require_positive_definite(metric: AbstractMetricField, fname: str, /) -> None:
    r"""Raise `NotImplementedError` unless *metric* is positive-definite.

    A Riemannian magnitude $\sqrt{v^\top G v}$ is real-valued only when $G$ is
    positive-definite.  Under an indefinite (pseudo-Riemannian) metric such as
    `~coordinax.manifolds.MinkowskiMetric`, $v^\top G v$ is *negative* for
    timelike vectors, so the square root evaluates to ``nan`` -- a wrong answer
    wearing the costume of a real one.  Callers get a loud failure instead.

    Parameters
    ----------
    metric
        The metric field to check, via its ``signature``.
    fname
        Name of the calling function, used in the error message.

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> from coordinax._src.manifolds._utils import require_positive_definite

    A Riemannian metric passes silently:

    >>> require_positive_definite(cxm.FlatMetric(3), "norm") is None
    True

    An indefinite one does not:

    >>> try:
    ...     require_positive_definite(cxm.MinkowskiMetric(), "norm")
    ... except NotImplementedError as e:
    ...     print(e)
    norm() supports only positive-definite metrics, but MinkowskiMetric has
    signature (-1, 1, 1, 1), which is pseudo-Riemannian (indefinite). Under such
    a metric `sqrt(v^T G v)` is `nan` for timelike vectors rather than a
    meaningful magnitude. Use `interval` for the signed square, or
    `proper_time` / `proper_distance` for the magnitude of a timelike /
    spacelike pair.

    """
    if all(s > 0 for s in metric.signature):
        return
    msg = (
        f"{fname}() supports only positive-definite metrics, but "
        f"{type(metric).__name__} has signature {tuple(metric.signature)}, which "
        "is pseudo-Riemannian (indefinite). Under such a metric "
        "`sqrt(v^T G v)` is `nan` for timelike vectors rather than a "
        "meaningful magnitude. Use `interval` for the signed square, or "
        "`proper_time` / `proper_distance` for the magnitude of a timelike / "
        "spacelike pair."
    )
    raise NotImplementedError(msg)


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


def raw_value(x: Any, /) -> Array:
    """Return the bare magnitude of *x*, ignoring any unit it carries.

    Only the *sign* of a contraction like ``g(v,v)`` is ever wanted from this,
    and the unit (``m2``, ``rad2/s2``, ...) does not bear on a sign. Note this
    *discards* the unit rather than converting it -- ``ustrip("")`` would raise
    on a dimensionful value.
    """
    return jnp.asarray(x.value if is_any_quantity(x) else x)
