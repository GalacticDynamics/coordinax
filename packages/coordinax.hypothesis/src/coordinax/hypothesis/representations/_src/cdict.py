"""Hypothesis strategies for CDict objects.

A CDict is a mapping from component-name strings to quantity-like array values.
This module provides strategies for generating valid CDict objects that match
chart component schemas.

"""

__all__ = ("cdicts",)

from typing import Any, cast

import hypothesis.strategies as st
import jax.numpy as jnp
import plum
from hypothesis.extra.array_api import make_strategies_namespace

import coordinax.charts as cxc
import coordinax.representations as cxr
from coordinax.internal.custom_types import CDict

from coordinax.hypothesis.utils import draw_if_strategy, strip_return_annotation

# Create array API strategies namespace for JAX
xps = make_strategies_namespace(jnp)


@plum.dispatch.multi(
    (st.SearchStrategy, st.SearchStrategy),
    (cxc.AbstractChart, st.SearchStrategy),
    (st.SearchStrategy, cxr.Representation),
)
@strip_return_annotation
@st.composite
def cdicts(  # noqa: F811
    draw: st.DrawFn,
    chart: cxc.AbstractChart | st.SearchStrategy,
    rep: cxr.Representation | st.SearchStrategy,
    /,
    **kwargs: Any,
) -> CDict:
    """Generate a CDict from chart/representation values or strategies.

    >>> import coordinax.hypothesis.main as cxst
    >>> from hypothesis import given
    >>> import coordinax.charts as cxc

    >>> @given(p=cxst.cdicts(cxc.cart3d, cxst.representations()))
    ... def test_cdict_with_rep_strategy(p):
    ...     assert set(p.keys()) == {"x", "y", "z"}

    """
    chart = draw_if_strategy(draw, chart)
    rep = draw_if_strategy(draw, rep)

    dtype = draw_if_strategy(draw, kwargs.pop("dtype", jnp.float32))
    shape = draw_if_strategy(draw, kwargs.pop("shape", ()))
    elements = draw_if_strategy(draw, kwargs.pop("elements", None))

    # Redispatch to the representation-specific implementation based on the chart
    strategy = cdicts(chart, rep, dtype=dtype, shape=shape, elements=elements, **kwargs)
    return draw(cast(st.SearchStrategy[CDict], strategy))


@plum.dispatch
@strip_return_annotation
@st.composite
def cdicts(  # noqa: F811
    draw: st.DrawFn,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kwargs: Any,
) -> CDict:
    """Generate a CDict for a chart and full ``Representation`` descriptor.

    >>> import coordinax.hypothesis.representations as cxsr
    >>> import coordinax.representations as cxr
    >>> import coordinax.charts as cxc
    >>> from hypothesis import given

    >>> @given(p=cxsr.cdicts(cxc.sph3d, cxr.point))
    ... def test_cdict_with_concrete_representation(p):
    ...     assert set(p.keys()) == {"r", "theta", "phi"}

    """
    # Break apart the rep and redispatch
    strategy = cdicts(chart, rep.geom_kind, rep.basis, rep.semantic_kind, **kwargs)
    return draw(cast(st.SearchStrategy[CDict], strategy))


@plum.dispatch
@strip_return_annotation
@st.composite
def cdicts(  # noqa: F811
    draw: st.DrawFn,
    chart: cxc.AbstractChart,
    geom_kind: cxr.PointGeometry,
    basis: cxr.AbstractBasis,
    semantic_kind: cxr.AbstractSemanticKind,
    /,
    **kwargs: Any,
) -> CDict:
    """Generate a point-geometry CDict with representation validity checks.

    >>> import coordinax.hypothesis.representations as cxsr
    >>> import coordinax.representations as cxr
    >>> import coordinax.charts as cxc
    >>> from hypothesis import given, strategies as st
    >>> import pytest

    >>> class FakeBasis(cxr.AbstractBasis):
    ...     pass

    >>> @given(data=st.data())
    ... def test_invalid_point_basis_raises(data):
    ...     with pytest.raises(TypeError, match="NoBasis"):
    ...         data.draw(cxsr.cdicts(cxc.cart3d, cxr.PointGeometry(), FakeBasis(), cxr.Location()))

    """
    if not isinstance(basis, cxr.NoBasis):
        raise TypeError("cdicts with PointGeometry must have NoBasis")
    if not isinstance(semantic_kind, cxr.Location):
        raise TypeError("cdicts with PointGeometry must have Location semantic kind")

    # The default implementation for cdicts is the point-geometry case, so we can safely redispatch to the rep-less methods.
    strategy = cdicts(chart, **kwargs)
    return draw(cast(st.SearchStrategy[CDict], strategy))
