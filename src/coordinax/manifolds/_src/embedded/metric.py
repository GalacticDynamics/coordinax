"""Representations for embedded manifolds."""

__all__ = ("InducedMetric",)

import dataclasses

from typing import final

import jax

import quaxed.numpy as qnp
import unxt as u

import coordinax.charts as cxc
from .embedmap import AbstractEmbeddingMap
from coordinax.internal import (
    QuantityMatrix,
    UnitsMatrix,
    cdict_units,
    pack_nonuniform_unit,
)
from coordinax.internal.custom_types import CDict, OptUSys
from coordinax.manifolds._src.base import AbstractMetric

DMLS = u.unit("")


def _jacobian_embed_map(
    embed_map: AbstractEmbeddingMap, at: CDict, usys: OptUSys
) -> QuantityMatrix:
    """Compute the Jacobian of ``embed_map`` at ``at`` as a ``QuantityMatrix``.

    Mirrors the general fallback of ``jac_pt_map`` but differentiates
    the embedding function instead of a chart transition map.

    Parameters
    ----------
    embed_map
        Embedding from intrinsic to ambient coordinates.
    at
        Base point in intrinsic coordinates.
    usys
        Optional unit system for the computation.

    Returns
    -------
    QuantityMatrix
        2-D ``QuantityMatrix`` of shape ``(n_ambient, n_intrinsic)`` where
        ``J.value[j, i] = \u2202(ambient_j) / \u2202(intrinsic_i)``.

    """
    embed_fn = embed_map.embed
    intrinsic_keys = embed_map.intrinsic.components
    ambient_keys = embed_map.ambient.components

    # Pack `at` → plain array + per-component from-units
    xat, ufrom = pack_nonuniform_unit(at, intrinsic_keys)

    # Run embedding once to determine output units
    at_ambient = embed_fn(at, usys=usys)
    uto = cdict_units(at_ambient, ambient_keys)

    # Replace None with dimensionless
    ufrom_ = tuple(uf if uf is not None else DMLS for uf in ufrom)
    uto_ = tuple(ut if ut is not None else DMLS for ut in uto)

    # Build (n_ambient × n_intrinsic) unit matrix
    unit_matrix = UnitsMatrix(tuple(tuple(tj / fi for fi in ufrom_) for tj in uto_))  # ty: ignore[unsupported-operator]

    # Plain-array embedding for jacfwd
    def embed_fn_arr(x_arr: qnp.ndarray) -> qnp.ndarray:
        q = {k: u.Q(x_arr[i], ufrom_[i]) for i, k in enumerate(intrinsic_keys)}
        out = embed_fn(q, usys=usys)
        vals = [
            u.ustrip(uto_[j], out[k])
            if isinstance(out[k], u.AbstractQuantity)
            else qnp.asarray(out[k])
            for j, k in enumerate(ambient_keys)
        ]
        return qnp.stack(vals)

    J_arr = jax.jacfwd(embed_fn_arr)(xat)  # shape (n_ambient, n_intrinsic)
    return QuantityMatrix(J_arr, unit=unit_matrix)


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class InducedMetric(AbstractMetric):
    r"""Pullback metric induced by an embedding map.

    Given an embedding $\iota : N \hookrightarrow M$, the metric $g_N$ on the
    submanifold is the pullback of the ambient metric $g_M$:

    $$g_N = \iota^* g_M, \quad \text{or component-wise}\quad
      (g_N)_{ij} = (J^T G J)_{ij},$$

    where $J = \partial \iota / \partial q$ is the Jacobian of the embedding
    map and $G = g_M$ is the ambient metric evaluated at $\iota(p)$.

    Parameters
    ----------
    embed_map : AbstractEmbeddingMap
        The embedding map from the submanifold into the ambient space.
    ambient_metric : AbstractMetric
        The Riemannian metric on the ambient manifold.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> embed_map = cxm.TwoSphereIn3D(radius=u.Q(1.0, "km"))
    >>> ambient_metric = cxm.EuclideanMetric(3)
    >>> M = cxm.InducedMetric(embed_map, ambient_metric)
    >>> at = {"theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0.0, "rad")}
    >>> M.metric_matrix(cxc.sph2, at=at)
    QuantityMatrix([[1., 0.],
                    [0., 1.]], '((km2 / rad2, km2 / rad2), (km2 / rad2, km2 / rad2))')

    Embedding into a Riemannian ambient space yields a Riemannian induced metric:

    >>> M.signature
    (1, 1)
    >>> M.ndim
    2

    """

    embed_map: AbstractEmbeddingMap
    ambient_metric: AbstractMetric

    @property
    def signature(self) -> tuple[int, ...]:
        """Metric signature: ``(1,) * m`` where ``m`` is the intrinsic dimension.

        Embedding into a Riemannian ambient manifold always produces a
        Riemannian induced metric (``J^T g_M J`` is positive-definite when
        ``J`` has full column rank).
        """
        return (1,) * self.embed_map.intrinsic.ndim

    def metric_matrix(
        self, _: cxc.AbstractChart, /, *, at: CDict, usys: OptUSys = None
    ) -> QuantityMatrix:
        r"""Compute the induced metric $g_N = J^T G_M J$.

        Computes the Jacobian of the embedding evaluated at ``at``, then
        contracts with the ambient metric at the embedded point.

        """
        # Compute Jacobian of the embedding as a QuantityMatrix:
        # J shape (n_ambient, n_intrinsic)
        J = _jacobian_embed_map(self.embed_map, at, usys=usys)

        # Evaluate embedding at base point to get ambient point
        at_ambient = self.embed_map.embed(at)

        # Ambient metric at the embedded point (also QuantityMatrix)
        ambient_chart = self.embed_map.ambient
        G_ambient = self.ambient_metric.metric_matrix(
            ambient_chart, at=at_ambient, usys=usys
        )

        JT = qnp.transpose(J, (1, 0))  # (n_intrinsic, n_ambient)
        return JT @ G_ambient @ J  # ty: ignore[invalid-return-type]
