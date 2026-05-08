"""Hypothesis strategies for coordinax vectors.

This module provides a multiple-dispatch strategy family built with
``plum.dispatch`` and ``hypothesis.strategies.composite``.

Dispatch structure (public positional argument signatures)::

    vectors()                          # draws chart from all charts
    vectors(chart: SearchStrategy)     # draws chart from strategy
    vectors(chart: AbstractChart)      # concrete chart; rep/manifold inferred
    vectors(chart, rep)                # chart + representation (each concrete
                                       # or strategy)
    vectors(chart, rep, manifold)      # chart + representation + manifold
                                       # (each concrete or strategy)

Any positional argument may be either a concrete instance *or* a
``hypothesis.strategies.SearchStrategy`` that produces one.  Strategy-valued
arguments are drawn first and the call is then re-dispatched to the concrete
implementation.

``**kwargs`` (``dtype``, ``shape``, ``elements``) are forwarded to the
underlying CDict generators.

"""

__all__ = ("vectors",)


from typing import Any, cast

import plum
from hypothesis import assume, strategies as st

import coordinax.charts as cxc
import coordinax.manifolds as cxm
import coordinax.representations as cxr
import coordinax.vectors as cxv

import coordinax.hypothesis.charts as cxcst
import coordinax.hypothesis.representations as cxrst
from coordinax.hypothesis.utils import draw_if_strategy, strip_return_annotation


@plum.dispatch.abstract
def vectors(*args: Any, **kwargs: Any) -> cxv.Point:
    """Generate valid ``coordinax.vectors.Point`` instances.

    This is the abstract entry point for the multiple-dispatch strategy family.
    Concrete overloads are registered below and selected by plum based on the
    runtime types of the positional arguments.

    Examples
    --------
    >>> import coordinax.hypothesis.vectors as cxvst
    >>> import coordinax.hypothesis.main as cxst
    >>> from hypothesis import given
    >>> import coordinax.vectors as cxv

    >>> @given(vec=cxvst.vectors())
    ... def test_any_vector(vec: cxv.Point) -> None:
    ...     assert isinstance(vec, cxv.Point)

    """
    raise NotImplementedError  # pragma: no cover


st.register_type_strategy(cxv.Point, lambda _: vectors())  # ty: ignore[missing-argument]


#####################################################################
# Strategy-argument dispatches (draw strategy args, then redispatch)
#####################################################################


@plum.dispatch
@strip_return_annotation
@st.composite
def vectors(  # noqa: F811
    draw: st.DrawFn,
    chart: st.SearchStrategy = cxcst.charts(exclude=(cxc.Time1D,)),  # ty: ignore[missing-argument],
    /,
    **kw: Any,
) -> cxv.Point:
    """Generate a vector by first drawing a chart from *chart*.

    >>> import coordinax.hypothesis.vectors as cxvst
    >>> import coordinax.hypothesis.charts as cxcst
    >>> import coordinax.charts as cxc
    >>> from hypothesis import given

    Draw from all registered charts (default):

    >>> @given(vec=cxvst.vectors())
    ... def test_all_charts(vec): ...

    Restrict to 3-D Cartesian charts only:

    >>> @given(vec=cxvst.vectors(cxcst.charts(filter=cxc.Abstract3D)))
    ... def test_3d_charts(vec): ...

    """
    chart = draw(chart)
    try:
        return draw(vectors(chart, **kw))  # ty: ignore[missing-argument]
    except (TypeError, ValueError, plum.NotFoundLookupError):
        assume(False)


@plum.dispatch.multi(
    (st.SearchStrategy, st.SearchStrategy),
    (cxc.AbstractChart, st.SearchStrategy),
    (st.SearchStrategy, cxr.Representation),
)
@strip_return_annotation
@st.composite
def vectors(  # noqa: F811
    draw: st.DrawFn,
    chart: cxc.AbstractChart | st.SearchStrategy,
    rep: cxr.Representation | st.SearchStrategy,
    /,
    **kw: Any,
) -> cxv.Point:
    """Generate a point after drawing strategy-valued *chart* and/or *rep*.

    This overload handles all cases where at least one of ``chart`` or ``rep``
    is a {class}`hypothesis.strategies.SearchStrategy`.  Both are resolved
    first, then the call is re-dispatched to the concrete ``(chart, rep)``
    overload.

    >>> import coordinax.hypothesis.vectors as cxvst
    >>> import coordinax.hypothesis.charts as cxcst
    >>> import coordinax.hypothesis.representations as cxrst
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> from hypothesis import given

    Chart strategy + concrete rep:

    >>> @given(vec=cxvst.vectors(cxcst.charts(), cxr.point))
    ... def test_chart_strat_rep(vec): ...

    Concrete chart + rep strategy:

    >>> @given(vec=cxvst.vectors(cxc.cart3d, cxrst.representations()))
    ... def test_chart_rep_strat(vec): ...

    """
    chart = draw_if_strategy(draw, chart)
    rep = draw_if_strategy(draw, rep)
    try:
        return draw(vectors(chart, rep, **kw))  # ty: ignore[missing-argument, too-many-positional-arguments]
    except (TypeError, ValueError, plum.NotFoundLookupError):
        assume(False)


@plum.dispatch.multi(
    (st.SearchStrategy, st.SearchStrategy, st.SearchStrategy),
    (cxc.AbstractChart, st.SearchStrategy, st.SearchStrategy),
    (st.SearchStrategy, cxr.Representation, st.SearchStrategy),
    (st.SearchStrategy, st.SearchStrategy, cxm.AbstractManifold),
    (cxc.AbstractChart, cxr.Representation, st.SearchStrategy),
    (cxc.AbstractChart, st.SearchStrategy, cxm.AbstractManifold),
    (st.SearchStrategy, cxr.Representation, cxm.AbstractManifold),
)
@strip_return_annotation
@st.composite
def vectors(  # noqa: F811
    draw: st.DrawFn,
    chart: cxc.AbstractChart | st.SearchStrategy,
    rep: cxr.Representation | st.SearchStrategy,
    manifold: cxm.AbstractManifold | st.SearchStrategy,
    /,
    **kw: Any,
) -> cxv.AbstractVector:
    """Generate a vector after drawing strategy-valued *chart*, *rep*, and/or *manifold*.

    Handles all three-argument combinations where at least one of ``chart``,
    ``rep``, or ``manifold`` is a strategy.  Concrete values are re-dispatched
    to the ``(chart, rep, manifold)`` concrete overload.

    Before redispatching, the drawn manifold (if not ``None``) is validated
    against the drawn chart; an incompatible pair raises
    {class}`ValueError`.

    Examples
    --------
    >>> import coordinax.hypothesis.vectors as cxvst
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.manifolds as cxm
    >>> from hypothesis import given
    >>> import hypothesis.strategies as st

    Explicit manifold via strategy:

    >>> manifold = cxm.EuclideanManifold(3)
    >>> @given(vec=cxvst.vectors(cxc.cart3d, cxr.point, st.just(manifold)))
    ... def test_manifold_strat(vec): ...

    """
    chart = draw_if_strategy(draw, chart)
    rep = draw_if_strategy(draw, rep)
    manifold = draw_if_strategy(draw, manifold)

    if manifold is not None and not manifold.has_chart(chart):
        raise ValueError(f"Manifold {manifold!r} does not support chart {chart!r}.")

    try:
        return draw(vectors(chart, rep, manifold, **kw))  # ty: ignore[too-many-positional-arguments]
    except (TypeError, ValueError, plum.NotFoundLookupError):
        assume(False)


#####################################################################
# Concrete-argument dispatches (terminal implementations)
#####################################################################


@plum.dispatch
@strip_return_annotation
@st.composite
def vectors(  # noqa: F811
    draw: st.DrawFn,
    chart: cxc.AbstractChart,
    /,
    **kw: Any,
) -> cxv.Point:
    """Generate a point for a concrete *chart*.

    Examples
    --------
    >>> import coordinax.hypothesis.vectors as cxvst
    >>> import coordinax.charts as cxc
    >>> from hypothesis import given

    >>> @given(vec=cxvst.vectors(cxc.cart3d))
    ... def test_cart3d(vec): ...

    """
    rep = draw(cxrst.representations(check_valid=True))
    try:
        return draw(vectors(chart, rep, **kw))  # ty: ignore[missing-argument, too-many-positional-arguments]
    except (TypeError, ValueError, plum.NotFoundLookupError):
        assume(False)


@plum.dispatch
@strip_return_annotation
@st.composite
def vectors(  # noqa: F811
    draw: st.DrawFn,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> cxv.Point:
    """Generate a point for a concrete *chart* and *rep* (manifold inferred).

    Examples
    --------
    >>> import coordinax.hypothesis.vectors as cxvst
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> from hypothesis import given

    >>> @given(vec=cxvst.vectors(cxc.cart3d, cxr.point))
    ... def test_cart3d_point(vec): ...

    """
    cdict = draw(cxrst.cdicts(chart, rep, **kw))  # ty: ignore[missing-argument]
    try:
        out = cxv.Point.from_(cdict, chart, rep)
    except plum.NotFoundLookupError as exc:
        raise ValueError(f"Could not infer a manifold for chart {chart!r}.") from exc
    return cast("cxv.Point", out)


@plum.dispatch
@strip_return_annotation
@st.composite
def vectors(  # noqa: F811
    draw: st.DrawFn,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    manifold: cxm.AbstractManifold,
    /,
    **kw: Any,
) -> cxv.AbstractVector:
    """Generate a vector for a fully-concrete ``(chart, rep, manifold)`` triple.

    This is the terminal overload.  All arguments are concrete instances and no
    further redispatch occurs.

    Examples
    --------
    >>> import coordinax.hypothesis.vectors as cxvst
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.manifolds as cxm
    >>> from hypothesis import given

    >>> manifold = cxm.EuclideanManifold(3)
    >>> @given(vec=cxvst.vectors(cxc.cart3d, cxr.point, manifold))
    ... def test_explicit_manifold(vec): ...

    """
    if not manifold.has_chart(chart):
        raise ValueError(f"Manifold {manifold!r} does not support chart {chart!r}.")

    data = draw(cxrst.cdicts(chart, rep, **kw))  # ty: ignore[missing-argument]
    return cxv.Point.from_(data, chart, rep, manifold)  # ty: ignore[invalid-return-type]
