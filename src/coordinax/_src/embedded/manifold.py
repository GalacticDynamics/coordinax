"""Representations for embedded manifolds."""

__all__ = ("EmbeddedManifold",)

import dataclasses

from typing import Any, Final, Generic, cast, final
from typing_extensions import override

import jax
import plum

import coordinax.api.charts as cxcapi
import coordinax.api.manifolds as cxmapi
from .embedmap import AbstractEmbeddingMap, AmbientT, IntrinsicT
from .metric import PullbackMetric
from coordinax._src.base import AbstractAtlas, AbstractChart, AbstractManifold
from coordinax._src.custom_types import CDict, OptUSys

UNSUPPORTED_CHART_MESSAGE: Final[str] = (
    "{0} chart {1} is not supported by the manifold's {0} atlas {2}."
)


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class EmbeddedManifold(AbstractManifold, Generic[IntrinsicT, AmbientT]):
    r"""Embedded manifold.

    Examples
    --------
    Embed/project `{class}`~coordinax.charts.SphericalTwoSphere` through an ambient
    `{class}`~coordinax.charts.Spherical3D` chart::

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u

    >>> M = cxm.EmbeddedManifold(
    ...     intrinsic=cxm.S2,
    ...     ambient=cxm.R3,
    ...     embed_map=cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> sph = cxm.pt_embed(p, M)
    >>> sph
    {'r': Q(2., 'km'), 'theta': Angle(1.57079633, 'rad'), 'phi': Angle(0., 'rad')}

    >>> p2 = cxm.pt_project(sph, M)
    >>> p2
    {'theta': Angle(1.57079633, 'rad'), 'phi': Angle(0., 'rad')}
    >>> jnp.allclose(p2["theta"].value, p["theta"].value)
    Array(True, dtype=bool)

    """

    intrinsic: AbstractManifold
    ambient: AbstractManifold
    embed_map: AbstractEmbeddingMap[IntrinsicT, AmbientT]

    # ===============================================================
    # Embedding API

    def embed(
        self,
        intrinsic_point: CDict,
        from_intrinsic_chart: AbstractChart[Any, Any, Any],
        to_ambient_chart: AbstractChart[Any, Any, Any],
        /,
        *,
        usys: OptUSys = None,
    ) -> CDict:
        out = cxmapi.pt_embed(
            intrinsic_point, from_intrinsic_chart, to_ambient_chart, self, usys=usys
        )
        return cast("CDict", out)

    def project(
        self,
        ambient_point: CDict,
        from_ambient_chart: AbstractChart[Any, Any, Any],
        to_intrinsic_chart: AbstractChart[Any, Any, Any],
        /,
        *,
        usys: OptUSys = None,
    ) -> CDict:
        out = cxmapi.pt_project(
            ambient_point, from_ambient_chart, to_intrinsic_chart, self, usys=usys
        )
        return cast("CDict", out)

    # ===============================================================
    # Manifold API

    @property
    def metric(self) -> PullbackMetric:
        """Induced (pullback) Riemannian metric from the ambient manifold."""
        return PullbackMetric(self.embed_map, self.ambient.metric)

    @override
    @property
    def atlas(self) -> AbstractAtlas:
        return self.intrinsic.atlas


# ===================================================================
# Point embedding


@plum.dispatch
def pt_embed(
    p_intrinsic: CDict, M: EmbeddedManifold, /, *, usys: OptUSys = None
) -> CDict:
    """Embed intrinsic point coordinates into ambient coordinates (manifold)."""
    # Redispatch to the more general pt_embed that handles chart transitions
    # in both the ambient and intrinsic charts.
    intrinsic_chart = M.embed_map.intrinsic
    ambient_chart = M.embed_map.ambient
    out = cxmapi.pt_embed(p_intrinsic, intrinsic_chart, ambient_chart, M, usys=usys)
    return cast("CDict", out)


@plum.dispatch
def pt_embed(
    p_intrinsic: CDict,
    from_intrinsic_chart: AbstractChart,
    to_ambient_chart: AbstractChart,
    M: EmbeddedManifold,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Embed intrinsic point coordinates into ambient coordinates (manifold)."""
    # Check the the intrinsic chart is in the manifold's intrinsic atlas
    if not M.intrinsic.has_chart(from_intrinsic_chart):
        raise ValueError(
            UNSUPPORTED_CHART_MESSAGE.format(
                "intrinsic", from_intrinsic_chart, M.intrinsic.atlas
            )
        )
    # Check that the ambient chart is supported by the manifold's ambient atlas
    if not M.ambient.has_chart(to_ambient_chart):
        raise ValueError(
            UNSUPPORTED_CHART_MESSAGE.format(
                "ambient", to_ambient_chart, M.ambient.atlas
            )
        )
    # Now that it's confirmed that the charts are compatible, we can dispatch to
    # the actual implementation that handles the embedding.
    out = cxmapi.pt_embed(
        p_intrinsic, from_intrinsic_chart, to_ambient_chart, M.embed_map, usys=usys
    )
    return cast("CDict", out)


@plum.dispatch
def pt_embed(
    p_intrinsic: CDict,
    from_intrinsic_chart: AbstractChart,
    to_ambient_chart: AbstractChart,
    embed_map: AbstractEmbeddingMap,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    r"""Embed intrinsic point coordinates into ambient coordinates."""
    # Step 1: point transition from intrinsic chart to embedding's intrinsic chart
    p_intrinsic_in_embedding_chart = cxcapi.pt_map(
        p_intrinsic, from_intrinsic_chart, embed_map.intrinsic, usys=usys
    )
    # Step 2: embed from embedding's intrinsic chart to ambient chart
    p_ambient = embed_map.embed(p_intrinsic_in_embedding_chart)
    # Step 3: point transition from embedding's ambient chart to target chart
    p_ambient_in_target_chart = cxcapi.pt_map(
        p_ambient, embed_map.ambient, to_ambient_chart, usys=usys
    )
    return cast("CDict", p_ambient_in_target_chart)


# ===================================================================
# Point projection


@plum.dispatch
def pt_project(
    p_ambient: CDict, M: EmbeddedManifold, /, *, usys: OptUSys = None
) -> CDict:
    """Project ambient coordinates onto intrinsic chart coordinates (manifold)."""
    # Redispatch to the more general pt_project that handles chart
    # transitions in both the ambient and intrinsic charts.
    out = cxmapi.pt_project(
        p_ambient, M.embed_map.ambient, M.embed_map.intrinsic, M, usys=usys
    )
    return cast("CDict", out)


@plum.dispatch
def pt_project(
    p_ambient: CDict,
    from_ambient_chart: AbstractChart,
    to_intrinsic_chart: AbstractChart,
    M: EmbeddedManifold,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Project ambient coordinates onto intrinsic chart coordinates (manifold)."""
    # Check the the ambient chart is in the manifold's ambient atlas
    if not M.ambient.has_chart(from_ambient_chart):
        raise ValueError(
            UNSUPPORTED_CHART_MESSAGE.format(
                "ambient", from_ambient_chart, M.ambient.atlas
            )
        )
    # Check that the intrinsic chart is supported by the manifold's intrinsic atlas
    if not M.intrinsic.has_chart(to_intrinsic_chart):
        raise ValueError(
            UNSUPPORTED_CHART_MESSAGE.format(
                "intrinsic", to_intrinsic_chart, M.intrinsic.atlas
            )
        )
    # Now that it's confirmed that the charts are compatible, we can dispatch to
    # the actual implementation that handles the projection.
    out = cxmapi.pt_project(
        p_ambient, from_ambient_chart, to_intrinsic_chart, M.embed_map, usys=usys
    )
    return cast("CDict", out)


@plum.dispatch
def pt_project(
    p_ambient: CDict,
    from_ambient_chart: AbstractChart,
    to_intrinsic_chart: AbstractChart,
    embed_map: AbstractEmbeddingMap,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    r"""Project ambient coordinates onto intrinsic chart coordinates."""
    # Step 1: point transition from ambient chart to embedding's ambient chart
    p_ambient_in_embedding_chart = cxcapi.pt_map(
        p_ambient, from_ambient_chart, embed_map.ambient, usys=usys
    )
    # Step 2: project from embedding's ambient chart to intrinsic chart
    p_intrinsic = embed_map.project(p_ambient_in_embedding_chart)
    # Step 3: point transition from embedding's intrinsic chart to target chart
    p_intrinsic_in_target_chart = cxcapi.pt_map(
        p_intrinsic, embed_map.intrinsic, to_intrinsic_chart, usys=usys
    )
    return cast("CDict", p_intrinsic_in_target_chart)
