"""Manifold definitions and manifold inference helpers."""

__all__: tuple[str, ...] = ()


from typing import Final, cast

import plum

import coordinax.api.manifolds as cxmapi
import coordinax.charts as cxc
from .manifold import EmbeddedManifold
from coordinax._src.manifolds.custom_types import CDict, OptUSys

AMBIGUOUS_CHART_POINT_REALIZATION_MAP_MSG: Final[str] = (
    "Ambiguous point realization map: {0}_chart={1} is present in both "
    "intrinsic={2} and ambient={3}."
)


@plum.dispatch
def pt_map(
    p: CDict,
    manifold: EmbeddedManifold,
    from_chart: cxc.AbstractChart,
    to_chart: cxc.AbstractChart,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Convert between embedded manifolds with a shared ambient space.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> manifold = cxm.embedded_twosphere(radius=u.Q(1, "kpc"))
    >>> manifold
    EmbeddedManifold(intrinsic=HyperSphericalManifold(...),
                     ambient=EuclideanManifold(ndim=3),
                     embed_map=TwoSphereIn3D(radius=Q(1, 'kpc')))
    >>> x_cart = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}

    >>> x_sph2 = cxc.pt_map(x_cart, manifold,
    ...                                    cxc.cart3d, cxc.loncoslat_sph2)
    >>> x_sph2
    {'lon_coslat': Q(0.66164791, 'rad'), 'lat': Q(53.3007748, 'deg')}

    >>> cxc.pt_map(x_sph2, manifold,
    ...            cxc.loncoslat_sph2, cxc.cart3d)
    {'x': Q(0.26726124, 'kpc'), 'y': Q(0.53452248, 'kpc'),
     'z': Q(0.80178373, 'kpc')}

    """
    # First check if the intrinsic and ambient manifolds are the same, in which
    # case we can just delegate to the chart-level transition map.
    if manifold.intrinsic == manifold.ambient:
        out = cxc.pt_map(p, from_chart, to_chart, manifold.intrinsic, usys=usys)
        return cast("CDict", out)

    # Now that we know the intrinsic and ambient manifolds are different, we can
    # check for ambiguity in whether the charts are for the intrinsic or ambient
    # manifold. If either chart is present in both the intrinsic and ambient
    # manifolds, then we can't disambiguate which realization map to use, so we
    # raise an error.
    if manifold.intrinsic.has_chart(from_chart) and manifold.ambient.has_chart(
        from_chart
    ):
        raise ValueError(
            AMBIGUOUS_CHART_POINT_REALIZATION_MAP_MSG.format(
                "from", from_chart, manifold.intrinsic, manifold.ambient
            )
        )
    if manifold.intrinsic.has_chart(to_chart) and manifold.ambient.has_chart(to_chart):
        raise ValueError(
            AMBIGUOUS_CHART_POINT_REALIZATION_MAP_MSG.format(
                "to", to_chart, manifold.intrinsic, manifold.ambient
            )
        )

    # Now we know the charts are unambiguously intrinsic or ambient. We can
    # figure out whether we're realizing from intrinsic -> ambient (embedding),
    # ambient -> intrinsic (projection).
    map_fn = (
        cxmapi.pt_embed
        if manifold.intrinsic.has_chart(from_chart)
        else cxmapi.pt_project
    )
    out = map_fn(p, from_chart, to_chart, manifold, usys=usys)

    return cast("CDict", out)
