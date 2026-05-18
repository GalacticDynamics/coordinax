"""Minkowski spacetime manifold."""

__all__ = ("MinkowskiMetric",)

import dataclasses

from typing import final

import jax

from coordinax._src.base import AbstractDiagonalMetricField


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class MinkowskiMetric(AbstractDiagonalMetricField):
    r"""Pseudo-Riemannian (Lorentzian) metric on Minkowski spacetime.

    In the canonical {class}`~coordinax.charts.MinkowskiCT` chart
    $(ct, x, y, z)$, the Minkowski metric is

    $$\eta = \operatorname{diag}(-1, 1, 1, 1),$$

    where the time coordinate $ct = c\,t$ absorbs the speed of light so that
    all four components carry the same unit (length). The line element is

    $$ds^2 = -(d(ct))^2 + dx^2 + dy^2 + dz^2.$$

    **Signature.** The metric is **pseudo-Riemannian** with Lorentzian
    signature $(-1, 1, 1, 1)$ meaning one negative and three positive
    eigenvalues (convention: "mostly plus").

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinax.api.manifolds import metric_matrix

    Canonical Cartesian spacetime chart:

    >>> m = cxm.MinkowskiMetric()
    >>> M = cxm.MinkowskiManifold()
    >>> at = {"ct": jnp.array(0.0), "x": jnp.array(0.0),
    ...       "y": jnp.array(0.0), "z": jnp.array(0.0)}
    >>> metric_matrix(M, at, cxc.minkowskict).diagonal
    Array([-1.,  1.,  1.,  1.], dtype=float64)

    The signature is Lorentzian (pseudo-Riemannian):

    >>> m.signature
    (-1, 1, 1, 1)

    >>> m.ndim
    4

    """

    @property
    def signature(self) -> tuple[int, ...]:
        """Metric signature: ``(-1, 1, 1, 1)`` — Lorentzian pseudo-Riemannian."""
        return (-1, 1, 1, 1)
