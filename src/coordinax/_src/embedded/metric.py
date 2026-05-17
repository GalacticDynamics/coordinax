"""Representations for embedded manifolds."""

__all__ = ("PullbackMetric",)

import dataclasses

from typing import final

import jax

import quaxed.numpy as qnp
import unxt as u

from .embedmap import AbstractEmbeddingMap
from coordinax._src.base import AbstractMetricField
from coordinax._src.custom_types import CDict, OptUSys
from coordinax.internal import (
    QMatrix,
    UnitsMatrix,
    cdict_units,
    pack_nonuniform_unit,
)

DMLS = u.unit("")


def _jacobian_embed_map(
    embed_map: AbstractEmbeddingMap, at: CDict, usys: OptUSys
) -> QMatrix:
    """Compute the Jacobian of ``embed_map`` at ``at`` as a ``QMatrix``.

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
    QMatrix
        2-D ``QMatrix`` of shape ``(n_ambient, n_intrinsic)`` where
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
    return QMatrix(J_arr, unit=unit_matrix)


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class PullbackMetric(AbstractMetricField):
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
    ambient_metric : AbstractMetricField
        The Riemannian metric on the ambient manifold.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.api.manifolds as cxmapi
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> embed_map = cxm.TwoSphereIn3D(radius=u.Q(1.0, "km"))
    >>> ambient_metric = cxm.FlatMetric(3)
    >>> M = cxm.PullbackMetric(embed_map, ambient_metric)
    >>> M.signature
    (1, 1)
    >>> M.ndim
    2

    The metric matrix is obtained via the dispatch API on an
    :class:`~coordinax.manifolds.EmbeddedManifold`:

    >>> M_emb = cxm.EmbeddedManifold(
    ...     intrinsic=cxm.S2, ambient=cxm.R3,
    ...     embed_map=cxm.TwoSphereIn3D(radius=u.Q(1.0, "km")),
    ... )
    >>> at = {"theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0.0, "rad")}
    >>> g = cxmapi.metric_matrix(M_emb, at, cxc.sph2)
    >>> g.matrix.value
    Array([[1., 0.],
           [0., 1.]], dtype=float64, weak_type=True)
    >>> g.matrix.unit[0, 0]
    Unit("km2 / rad2")

    """

    embed_map: AbstractEmbeddingMap
    ambient_metric: AbstractMetricField

    @property
    def signature(self) -> tuple[int, ...]:
        """Metric signature: ``(1,) * m`` where ``m`` is the intrinsic dimension.

        Embedding into a Riemannian ambient manifold always produces a
        Riemannian induced metric (``J^T g_M J`` is positive-definite when
        ``J`` has full column rank).
        """
        return (1,) * self.embed_map.intrinsic.ndim
