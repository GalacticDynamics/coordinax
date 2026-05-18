"""Chart strategy extensions owned by the manifolds package."""

import hypothesis.strategies as st
import plum
from hypothesis import assume

import coordinax.manifolds as cxm

from coordinax.hypothesis.utils import draw_if_strategy, strip_return_annotation


@plum.dispatch
@strip_return_annotation
@st.composite
def charts(
    draw: st.DrawFn,
    chart_cls: type[cxm.EmbeddedChart],
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: int | None | st.SearchStrategy = None,
) -> cxm.EmbeddedChart:
    """Generate ``EmbeddedChart`` instances backed by ``TwoSphereIn3D``.

    The generated chart is always intrinsic 2-D, so ``ndim`` constraints
    different from ``2`` are discarded via ``hypothesis.assume``.
    """
    if filter or exclude:
        raise ValueError(
            "When chart_cls is provided, filter and exclude must be empty."
        )

    target_ndim = draw_if_strategy(draw, ndim)
    if target_ndim is not None and target_ndim != 2:
        assume(False)

    radius = draw(
        st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False)
    )
    return cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=radius))
