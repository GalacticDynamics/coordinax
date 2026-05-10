"""Customs manifolds."""

__all__ = ("CustomMetric",)

import dataclasses

from collections.abc import Callable
from typing import Any, final

import jax

from coordinax._src.base_metric import AbstractMetric


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class CustomMetric(AbstractMetric):  # ty: ignore[abstract-method-in-final-class]
    r"""Metric for a {class}`CustomManifold`, defined by user-provided callables.

    ``CustomMetric`` allows users to supply their own metric without subclassing
    ``AbstractMetric``. Both the metric-matrix callable and the signature must
    be provided at construction time.

    Parameters
    ----------
    metric_matrix : callable
        Callable ``(chart, /, *, at) -> Array`` returning the $(n \times n)$
        metric matrix at the given base point.
    signature : tuple[int, ...]
        Metric signature as a length-$n$ tuple of ``+1`` (positive eigenvalue)
        and ``-1`` (negative eigenvalue) entries. Use ``(1,) * n`` for a
        Riemannian metric of dimension $n$.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> def flat_3d(chart, /, *, at):
    ...     return jnp.eye(3)

    >>> atlas = cxm.CustomAtlas(
    ...     charts=(cxc.Cart3D,),
    ...     chart_default=cxc.cart3d,
    ... )
    >>> metric = cxm.CustomMetric(metric_matrix=flat_3d, signature=(1, 1, 1))
    >>> metric.signature
    (1, 1, 1)
    >>> metric.ndim
    3

    """

    metric_matrix: Callable[..., Any]
    """Callable ``(chart, /, *, at) -> Array`` returning the metric matrix."""

    signature: tuple[int, ...]
    """Metric signature as a length-n tuple of ``+1``/``-1`` per eigenvalue."""
