"""Manifold definitions and manifold inference helpers."""

__all__: tuple[str, ...] = ()


from typing import Final, cast

import plum

import coordinax.api.charts as cxcapi
import coordinax.api.manifolds as cxmapi
from .manifold import EmbeddedManifold
from coordinax._src.base import AbstractChart
from coordinax._src.custom_types import CDict, OptUSys

AMBIGUOUS_CHART_POINT_REALIZATION_MAP_MSG: Final[str] = (
    "Ambiguous point realization map: {0}_chart={1} is present in both "
    "intrinsic={2} and ambient={3}."
)


@plum.dispatch
def pt_map(
    p: CDict,
    M: EmbeddedManifold,
    from_chart: AbstractChart,
    to_chart: AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Convert between embedded manifolds with a shared ambient space.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> M = cxm.embedded_twosphere(radius=u.Q(1, "kpc"))
    >>> M
    EmbeddedManifold(intrinsic=HyperSphericalManifold(...),
                     ambient=Rn(3),
                     embed_map=TwoSphereIn3D(radius=Q(1, 'kpc')))
    >>> x_cart = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}

    >>> x_sph2 = cxc.pt_map(x_cart, M, cxc.cart3d, cxc.loncoslat_sph2)
    >>> x_sph2
    {'lon_coslat': Q(0.66164791, 'rad'), 'lat': Q(53.3007748, 'deg')}

    >>> cxc.pt_map(x_sph2, M, cxc.loncoslat_sph2, cxc.cart3d)
    {'x': Q(0.26726124, 'kpc'), 'y': Q(0.53452248, 'kpc'),
     'z': Q(0.80178373, 'kpc')}

    """
    # First check if the intrinsic and ambient manifolds are the same, in which
    # case we can just delegate to the chart-level transition map.
    if M.intrinsic == M.ambient:
        out = cxcapi.pt_map(p, from_chart, to_chart, usys=usys)
        return cast("CDict", out)

    # Now that we know the intrinsic and ambient manifolds are different, we can
    # check for ambiguity in whether the charts are for the intrinsic or ambient
    # manifold. If either chart is present in both the intrinsic and ambient
    # manifolds, then we can't disambiguate which realization map to use, so we
    # raise an error.
    if M.intrinsic.has_chart(from_chart) and M.ambient.has_chart(from_chart):
        raise ValueError(
            AMBIGUOUS_CHART_POINT_REALIZATION_MAP_MSG.format(
                "from", from_chart, M.intrinsic, M.ambient
            )
        )
    if M.intrinsic.has_chart(to_chart) and M.ambient.has_chart(to_chart):
        raise ValueError(
            AMBIGUOUS_CHART_POINT_REALIZATION_MAP_MSG.format(
                "to", to_chart, M.intrinsic, M.ambient
            )
        )

    # Now we know the charts are unambiguously intrinsic or ambient. We can
    # figure out whether we're realizing from intrinsic -> ambient (embedding),
    # ambient -> intrinsic (projection).
    map_fn = cxmapi.pt_embed if M.intrinsic.has_chart(from_chart) else cxmapi.pt_project
    out = map_fn(p, from_chart, to_chart, M, usys=usys)

    return cast("CDict", out)
