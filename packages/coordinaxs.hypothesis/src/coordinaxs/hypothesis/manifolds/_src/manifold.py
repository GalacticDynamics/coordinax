"""Hypothesis strategies for coordinax manifolds."""

__all__ = ("manifold_classes", "manifolds")

from collections.abc import Callable
from typing import Any, Final, cast

import hypothesis.strategies as st
import plum
from hypothesis import assume

import coordinax.charts as cxc
import coordinax.manifolds as cxm

from . import atlas as atlas_strategies
from ._common import matching_chart_classes_for_ndim
from coordinaxs.hypothesis.utils import (
    draw_if_strategy,
    get_all_subclasses,
    strip_return_annotation,
)


@st.composite
def manifold_classes(
    draw: st.DrawFn,
    /,
    filter: type
    | tuple[type, ...]
    | st.SearchStrategy[type | tuple[type, ...]] = object,
    *,
    exclude_abstract: bool | st.SearchStrategy[bool] = True,
    exclude: tuple[type, ...] = (),
) -> type[Any]:
    """Draw manifold classes (not instances) from concrete manifold subclasses."""
    classes = get_all_subclasses(
        cxm.AbstractManifold,
        filter=draw_if_strategy(draw, filter),
        exclude_abstract=draw_if_strategy(draw, exclude_abstract),
        exclude=exclude,
    )
    return cast("type[Any]", draw(st.sampled_from(classes)))


# ---------------------------------------------------------------------------
# ndim-compatibility helper — a new manifold type needs an entry only when no
# existing entry's base already covers it.
# ---------------------------------------------------------------------------

#: Which dimensionalities each concrete manifold type can be drawn at.
#:
#: This table is also the record of which types are drawable at all. Matching is
#: by `issubclass`, so a type is drawable when it is a subclass of some entry's
#: base -- it need not be listed in its own right, and a new subclass of a listed
#: base inherits that entry (and must therefore be served by the same dispatch).
#: A type matching no entry has no `manifolds` dispatch, so every draw of one is
#: discarded.
#:
#: A plain table rather than `plum.dispatch`: these signatures are all
#: ``type[X]``, which plum cannot treat as faithful, so its method cache was
#: disabled and every call ran a full resolution -- 5555 of them per 200
#: examples, because the no-argument dispatch tests the predicate against every
#: candidate class on every draw and product manifolds recurse. Dispatch bought
#: nothing here: the types are closed and local, and `issubclass` gives the same
#: subtype fallthrough. The listed types are mutually disjoint, so order is not
#: load-bearing; keep the most specific first anyway for anything added later.
_NDIM_SUPPORT: Final[
    tuple[tuple[type[cxm.AbstractManifold], Callable[[int], bool]], ...]
] = (
    # HyperSphericalManifold and EmbeddedManifold are always 2-D.
    (cxm.HyperSphericalManifold, lambda ndim: ndim == 2),
    (cxm.EmbeddedManifold, lambda ndim: ndim == 2),
    # A product needs at least one factor, each contributing >= 1 dimension.
    (cxm.CartesianProductManifold, lambda ndim: ndim >= 1),
    # CustomManifold works only where matching zero-arg charts exist.
    (cxm.CustomManifold, lambda ndim: bool(matching_chart_classes_for_ndim(ndim))),
    # EuclideanManifold supports any dimensionality.
    (cxm.EuclideanManifold, lambda _: True),
)


def _manifold_class_supports_ndim(
    cls: type[cxm.AbstractManifold], ndim: int | None, /
) -> bool:
    """Whether *cls* can be drawn, at *ndim* when one is requested.

    Types matching no `_NDIM_SUPPORT` base (``NoManifold``, ``MinkowskiManifold``)
    are not drawable at any dimensionality: no `manifolds` dispatch is
    registered for them, so selecting one only leads to the redispatch finding
    an empty candidate pool and discarding the example. The catch-all used to
    answer `True`, which put them in the pool on every draw and threw the
    resulting examples away -- for ``ndim=5``, where they are two of the three
    candidates, that was enough filtering to trip the ``filter_too_much``
    health check.
    """
    for base, supports in _NDIM_SUPPORT:
        if issubclass(cls, base):
            return ndim is None or supports(ndim)
    return False


# ---------------------------------------------------------------------------


@plum.dispatch.abstract
def manifolds(
    draw: st.DrawFn,
    manifold_cls: Any,
    /,
    filter: type | tuple[type, ...] | st.SearchStrategy = object,
    *,
    exclude: tuple[type, ...] = (),
    ndim: int | st.SearchStrategy[int] | None = None,
    required_chart_classes: tuple[type[cxc.AbstractChart], ...] = (),
) -> Any:
    """Generate manifold instances across the concrete manifold hierarchy.

    Parameters
    ----------
    draw
        The draw function used by the hypothesis composite strategy.
        Automatically provided by hypothesis.
    manifold_cls
        The manifold class to draw an instance of. If provided, the strategy
        draws an instance of this class. If None, the strategy draws an
        instance of any manifold class that satisfies filter and exclude.
    filter
        A class or tuple of classes to limit manifold classes to. Tuple filters
        use AND semantics. Strategy-valued filters are supported.
    exclude
        Specific classes to exclude.
    ndim
        ``manifold.ndim`` constraint. Can be ``None``, ``int``, or strategy.
    required_chart_classes
        Additional constraint used when drawing ``CustomManifold``. Forwarded to
        custom atlas generation.

    Returns
    -------
    coordinax.manifolds.AbstractManifold
        An instance of a concrete manifold class.

    Raises
    ------
    NotImplementedError
        If no strategy is registered for the selected manifold class.
    ValueError
        If ``manifold_cls`` is provided and ``filter``/``exclude`` are non-empty,
        or if ``required_chart_classes`` is passed for a non-custom manifold.

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch
@strip_return_annotation
@st.composite
def manifolds(
    draw: st.DrawFn,
    manifold_cls: None = None,
    /,
    filter: type | tuple[type, ...] | st.SearchStrategy = object,
    *,
    exclude: tuple[type, ...] = (),
    ndim: int | st.SearchStrategy[int] | None = None,
    required_chart_classes: tuple[type[cxc.AbstractChart], ...] = (),
) -> cxm.AbstractManifold:
    """Strategy to determine and draw manifold instances."""
    target_ndim = None if ndim is None else draw_if_strategy(draw, ndim)
    chosen_filter = draw_if_strategy(draw, filter)
    classes = tuple(
        cls
        for cls in get_all_subclasses(
            cxm.AbstractManifold,
            filter=chosen_filter,
            exclude_abstract=True,
            exclude=exclude,
        )
        if _manifold_class_supports_ndim(cls, target_ndim)
    )
    if not classes:
        assume(False)

    selected_cls = cast("type[cxm.AbstractManifold]", draw(st.sampled_from(classes)))
    kwargs: dict[str, Any] = {"ndim": target_ndim}
    if issubclass(selected_cls, cxm.CustomManifold):
        kwargs["required_chart_classes"] = required_chart_classes
    return draw(cast("Any", manifolds)(selected_cls, **kwargs))


@plum.dispatch
@strip_return_annotation
@st.composite
def manifolds(
    draw: st.DrawFn,
    manifold_cls: st.SearchStrategy,
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: Any = None,
    required_chart_classes: tuple[type[cxc.AbstractChart], ...] = (),
) -> Any:
    """Draw manifold classes from strategy-valued selectors and redispatch."""
    if filter or exclude:
        raise ValueError(
            "When manifold_cls is provided, filter and exclude must be empty."
        )
    if ndim is not None:
        raise ValueError("When manifold_cls is provided, ndim must be None.")

    manifold_cls = draw(manifold_cls)
    return draw(
        cast("Any", manifolds)(
            manifold_cls, ndim=ndim, required_chart_classes=required_chart_classes
        )
    )


@plum.dispatch
@strip_return_annotation
@st.composite
def manifolds(
    draw: st.DrawFn,
    manifold_cls: type[cxm.AbstractManifold],
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: int | st.SearchStrategy | None = None,
    required_chart_classes: tuple[type[cxc.AbstractChart], ...] = (),
) -> Any:
    """Draw any concrete manifold that is a subclass of an abstract manifold class.

    The abstract class is used as a ``filter`` and the call redispatches to
    the no-argument dispatch. ``filter`` and ``exclude`` must be empty.

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> import coordinaxs.hypothesis.manifolds as cxmst

    >>> from_abstract = cxmst.manifolds(cxm.AbstractManifold)

    """
    if filter or exclude:
        raise ValueError(
            "When manifold_cls is provided, filter and exclude must be empty."
        )

    return draw(
        cast("Any", manifolds)(
            filter=manifold_cls,
            exclude=(),
            ndim=ndim,
            required_chart_classes=required_chart_classes,
        )
    )


@plum.dispatch
@strip_return_annotation
@st.composite
def manifolds(
    draw: st.DrawFn,
    M_cls: type[cxm.EuclideanManifold],
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: int | st.SearchStrategy | None = None,
) -> Any:
    """Draw a ``EuclideanManifold`` of any dimensionality in ``[0, 4]``.

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> import coordinaxs.hypothesis.manifolds as cxmst

    >>> euclidean = cxmst.manifolds(cxm.EuclideanManifold)
    >>> euclidean_3d = cxmst.manifolds(cxm.EuclideanManifold, ndim=3)

    """
    target_ndim = draw_if_strategy(draw, ndim)
    dim = (
        draw(st.integers(min_value=0, max_value=4))
        if target_ndim is None
        else target_ndim
    )
    return M_cls(dim)


@plum.dispatch
@strip_return_annotation
@st.composite
def manifolds(
    draw: st.DrawFn,
    M_cls: type[cxm.HyperSphericalManifold],
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: int | st.SearchStrategy | None = None,
) -> Any:
    """Draw a ``HyperSphericalManifold`` (always 2-D).

    Examples with ``ndim != 2`` are discarded via ``hypothesis.assume``.

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> import coordinaxs.hypothesis.manifolds as cxmst

    >>> sphere = cxmst.manifolds(cxm.HyperSphericalManifold)

    """
    del M_cls
    target_ndim = draw_if_strategy(draw, ndim)
    if target_ndim is not None and target_ndim != 2:
        assume(False)
    return cxm.S2


@plum.dispatch
@strip_return_annotation
@st.composite
def manifolds(
    draw: st.DrawFn,
    M_cls: type[cxm.EmbeddedManifold],
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: int | st.SearchStrategy | None = None,
) -> Any:
    """Draw an ``EmbeddedManifold`` instance.

    Currently this strategy generates an embedded two-sphere by constructing
    ``EmbeddedManifold`` directly with:

    - ``intrinsic=S2``
    - ``ambient=R3``
    - ``embed_map=TwoSphereIn3D(radius=...)``

    Examples with ``ndim != 2`` are discarded via ``hypothesis.assume``.

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> import coordinaxs.hypothesis.manifolds as cxmst

    >>> embedded = cxmst.manifolds(cxm.EmbeddedManifold)

    """
    del M_cls
    target_ndim = draw_if_strategy(draw, ndim)
    if target_ndim is not None and target_ndim != 2:
        assume(False)

    radius = draw(
        st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False)
    )
    return cxm.EmbeddedManifold(
        intrinsic=cxm.S2, ambient=cxm.R3, embed_map=cxm.TwoSphereIn3D(radius=radius)
    )


@plum.dispatch
@strip_return_annotation
@st.composite
def manifolds(
    draw: st.DrawFn,
    manifold_cls: type[cxm.CustomManifold],
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: int | st.SearchStrategy | None = None,
    required_chart_classes: tuple[type[cxc.AbstractChart], ...] = (),
) -> Any:
    """Draw a ``CustomManifold`` backed by a drawn ``CustomAtlas``.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import coordinaxs.hypothesis.manifolds as cxmst

    >>> custom = cxmst.manifolds(cxm.CustomManifold)
    >>> custom_3d = cxmst.manifolds(cxm.CustomManifold, ndim=3)
    >>> custom_with_cart = cxmst.manifolds(
    ...     cxm.CustomManifold,
    ...     required_chart_classes=(cxc.Cart3D,),
    ...     ndim=3,
    ... )

    """
    target_ndim = draw_if_strategy(draw, ndim)
    atlas = draw(
        cast("Any", atlas_strategies.atlases)(
            cxm.CustomAtlas,
            ndim=target_ndim,
            required_chart_classes=required_chart_classes,
        )
    )
    metric = cxm.FlatMetric(atlas.ndim)
    return cxm.CustomManifold(atlas=atlas, metric=metric)


@plum.dispatch
@strip_return_annotation
@st.composite
def manifolds(
    draw: st.DrawFn,
    manifold_cls: type[cxm.CartesianProductManifold],
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: int | st.SearchStrategy | None = None,
) -> Any:
    """Draw a ``CartesianProductManifold`` with 1-5 non-product factor manifolds.

    The number of factors is drawn uniformly from 1 to 5. The total
    dimensionality of the product equals the sum of the factor dimensionalities.
    When ``ndim`` is given, each factor contributes at least 1 dimension, so the
    factor count is drawn from ``[1, min(5, ndim)]``; ``ndim < 1`` admits no
    product at all and is discarded via ``hypothesis.assume``.

    >>> import coordinax.manifolds as cxm
    >>> import coordinaxs.hypothesis.manifolds as cxmst

    >>> # Draw any CartesianProductManifold (1-5 factors)
    >>> product = cxmst.manifolds(cxm.CartesianProductManifold)

    >>> # Draw a CartesianProductManifold with total ndim=4
    >>> product_4d = cxmst.manifolds(cxm.CartesianProductManifold, ndim=4)

    """
    target_ndim = draw_if_strategy(draw, ndim)

    # Each factor needs at least 1 dimension, so bound the factor count by the
    # target up front rather than drawing 1-5 and rejecting the infeasible ones:
    # at `ndim=1` that rejected four draws in five.
    if target_ndim is None:
        n_factors = draw(st.integers(min_value=1, max_value=5))
        total_ndim = draw(st.integers(min_value=n_factors, max_value=n_factors + 4))
    else:
        # No factor count works below 1 dimension; that request is unsatisfiable.
        assume(target_ndim >= 1)
        n_factors = draw(st.integers(min_value=1, max_value=min(5, target_ndim)))
        total_ndim = target_ndim

    # Partition total_ndim into n_factors positive integers, drawing each factor
    # from the range that still leaves >= 1 dimension for every factor after it.
    # The previous form drew n_factors-1 *unique* cuts from `integers(1,
    # total_ndim - 1)`; when total_ndim == n_factors that range holds exactly as
    # many values as the list needs, so hypothesis had to rejection-sample its
    # way to the single valid answer -- and total_ndim == n_factors is the
    # common case whenever `ndim=` is pinned.
    dims: list[int] = []
    remaining = total_ndim
    for i in range(n_factors - 1):
        # Reserve >= 1 dimension for each factor still to be assigned after i.
        dim_i = draw(st.integers(1, remaining - (n_factors - i - 1)))
        dims.append(dim_i)
        remaining -= dim_i
    dims.append(remaining)  # last factor takes the rest, which is >= 1

    factors = tuple(
        draw(cast("Any", manifolds)(exclude=(cxm.CartesianProductManifold,), ndim=d))
        for d in dims
    )
    factor_names = tuple(f"f{i}" for i in range(n_factors))
    return cxm.CartesianProductManifold(factors=factors, factor_names=factor_names)
