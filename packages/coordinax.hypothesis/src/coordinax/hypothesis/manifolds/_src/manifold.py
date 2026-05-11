"""Hypothesis strategies for coordinax manifolds."""

__all__ = ("manifold_classes", "manifolds")

from typing import Any, cast

import hypothesis.strategies as st
import plum
from hypothesis import assume

import coordinax.charts as cxc
import coordinax.manifolds as cxm

from . import atlas as atlas_strategies
from coordinax.hypothesis.utils import (
    draw_if_strategy,
    get_all_subclasses,
    strip_return_annotation,
)


def _is_zero_arg_constructible(chart_cls: type[cxc.AbstractChart[Any, Any]], /) -> bool:
    """Return True if chart_cls can be instantiated with no arguments."""
    try:
        chart_cls()
    except TypeError:
        return False
    return True


def _matching_chart_classes_for_ndim(
    ndim: int, /
) -> tuple[type[cxc.AbstractChart[Any, Any]], ...]:
    """Return zero-arg chart classes with default instance ndim == target ndim."""
    classes: list[type[cxc.AbstractChart[Any, Any]]] = []
    for cls in get_all_subclasses(cxc.AbstractChart, exclude_abstract=True):
        cls = cast(type[cxc.AbstractChart[Any, Any]], cls)
        if not _is_zero_arg_constructible(cls):
            continue
        if cls().ndim == ndim:
            classes.append(cls)
    return tuple(classes)


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
    return cast(type[Any], draw(st.sampled_from(classes)))


# ---------------------------------------------------------------------------
# ndim-compatibility helper — extend by adding a new dispatch for each new
# concrete manifold type.
# ---------------------------------------------------------------------------


@plum.dispatch
def _manifold_class_supports_ndim(
    cls: type[cxm.EuclideanManifold], ndim: int, /
) -> bool:
    """EuclideanManifold supports any dimensionality."""
    return True


@plum.dispatch
def _manifold_class_supports_ndim(
    cls: type[cxm.HyperSphericalManifold], ndim: int, /
) -> bool:
    """HyperSphericalManifold is always 2-D."""
    return ndim == 2


@plum.dispatch
def _manifold_class_supports_ndim(
    cls: type[cxm.EmbeddedManifold], ndim: int, /
) -> bool:
    """EmbeddedManifold: only the 2-D embedded two-sphere is currently generated."""
    return ndim == 2


@plum.dispatch
def _manifold_class_supports_ndim(
    cls: type[cxm.CartesianProductManifold], ndim: int, /
) -> bool:
    """CartesianProductManifold requires at least 1 dimension."""
    return ndim >= 1


@plum.dispatch
def _manifold_class_supports_ndim(cls: type[cxm.CustomManifold], ndim: int, /) -> bool:
    """CustomManifold supports ndim when matching zero-arg charts exist."""
    return len(_matching_chart_classes_for_ndim(ndim)) > 0


@plum.dispatch
def _manifold_class_supports_ndim(
    cls: type[cxm.AbstractManifold], ndim: int, /
) -> bool:
    """Fallback: unknown manifold types are assumed to support any ndim."""
    return True


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
def manifolds(  # noqa: F811
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
        if target_ndim is None
        or cast(Any, _manifold_class_supports_ndim)(cls, target_ndim)
    )
    if not classes:
        assume(False)

    selected_cls = cast(type[cxm.AbstractManifold], draw(st.sampled_from(classes)))
    kwargs: dict[str, Any] = {"ndim": target_ndim}
    if issubclass(selected_cls, cxm.CustomManifold):
        kwargs["required_chart_classes"] = required_chart_classes
    return draw(cast(Any, manifolds)(selected_cls, **kwargs))


@plum.dispatch
@strip_return_annotation
@st.composite
def manifolds(  # noqa: F811
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
        cast(Any, manifolds)(
            manifold_cls,
            ndim=ndim,
            required_chart_classes=required_chart_classes,
        )
    )


@plum.dispatch
@strip_return_annotation
@st.composite
def manifolds(  # noqa: F811
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
    >>> import coordinax.hypothesis.manifolds as cxmst

    >>> from_abstract = cxmst.manifolds(cxm.AbstractManifold)

    """
    if filter or exclude:
        raise ValueError(
            "When manifold_cls is provided, filter and exclude must be empty."
        )

    return draw(
        cast(Any, manifolds)(
            filter=manifold_cls,
            exclude=(),
            ndim=ndim,
            required_chart_classes=required_chart_classes,
        )
    )


@plum.dispatch
@strip_return_annotation
@st.composite
def manifolds(  # noqa: F811
    draw: st.DrawFn,
    manifold_cls: type[cxm.EuclideanManifold],
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
    >>> import coordinax.hypothesis.manifolds as cxmst

    >>> euclidean = cxmst.manifolds(cxm.EuclideanManifold)
    >>> euclidean_3d = cxmst.manifolds(cxm.EuclideanManifold, ndim=3)

    """
    target_ndim = draw_if_strategy(draw, ndim)
    dim = (
        draw(st.integers(min_value=0, max_value=4))
        if target_ndim is None
        else target_ndim
    )
    return cxm.EuclideanManifold(dim)


@plum.dispatch
@strip_return_annotation
@st.composite
def manifolds(  # noqa: F811
    draw: st.DrawFn,
    manifold_cls: type[cxm.HyperSphericalManifold],
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
    >>> import coordinax.hypothesis.manifolds as cxmst

    >>> sphere = cxmst.manifolds(cxm.HyperSphericalManifold)

    """
    target_ndim = draw_if_strategy(draw, ndim)
    if target_ndim is not None and target_ndim != 2:
        assume(False)
    return cxm.HyperSphericalManifold()


@plum.dispatch
@strip_return_annotation
@st.composite
def manifolds(  # noqa: F811
    draw: st.DrawFn,
    manifold_cls: type[cxm.EmbeddedManifold],
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: int | st.SearchStrategy | None = None,
) -> Any:
    """Draw an ``EmbeddedManifold`` instance.

    Currently this strategy generates an embedded two-sphere by constructing
    ``EmbeddedManifold`` directly with:

    - ``intrinsic=HyperSphericalManifold()``
    - ``ambient=EuclideanManifold(3)``
    - ``embed_map=TwoSphereIn3D(radius=...)``

    Examples with ``ndim != 2`` are discarded via ``hypothesis.assume``.

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.hypothesis.manifolds as cxmst

    >>> embedded = cxmst.manifolds(cxm.EmbeddedManifold)

    """
    target_ndim = draw_if_strategy(draw, ndim)
    if target_ndim is not None and target_ndim != 2:
        assume(False)

    radius = draw(
        st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False)
    )
    return cxm.EmbeddedManifold(
        intrinsic=cxm.twosphere,
        ambient=cxm.euclidean3d,
        embed_map=cxm.TwoSphereIn3D(radius=radius),
    )


@plum.dispatch
@strip_return_annotation
@st.composite
def manifolds(  # noqa: F811
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
    >>> import coordinax.hypothesis.manifolds as cxmst

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
        cast(Any, atlas_strategies.atlases)(
            cxm.CustomAtlas,
            ndim=target_ndim,
            required_chart_classes=required_chart_classes,
        )
    )
    metric = cxm.EuclideanMetric(atlas.ndim)
    return cxm.CustomManifold(atlas=atlas, metric=metric)


@plum.dispatch
@strip_return_annotation
@st.composite
def manifolds(  # noqa: F811
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
    When ``ndim`` is given it must be at least the number of factors (each
    factor contributes at least 1 dimension); examples that cannot satisfy this
    are discarded via ``hypothesis.assume``.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.hypothesis.manifolds as cxmst

    >>> # Draw any CartesianProductManifold (1-5 factors)
    >>> product = cxmst.manifolds(cxm.CartesianProductManifold)

    >>> # Draw a CartesianProductManifold with total ndim=4
    >>> product_4d = cxmst.manifolds(cxm.CartesianProductManifold, ndim=4)

    """
    target_ndim = draw_if_strategy(draw, ndim)

    # Draw the number of factors: 1–5
    n_factors = draw(st.integers(min_value=1, max_value=5))

    # Each factor needs at least 1 dimension, so total_ndim >= n_factors.
    if target_ndim is not None:
        assume(target_ndim >= n_factors)
        total_ndim = target_ndim
    else:
        total_ndim = draw(st.integers(min_value=n_factors, max_value=n_factors + 4))

    # Partition total_ndim into n_factors positive integers via n_factors-1 cuts.
    cuts = (
        sorted(
            draw(
                st.lists(
                    st.integers(1, total_ndim - 1),
                    min_size=n_factors - 1,
                    max_size=n_factors - 1,
                    unique=True,
                )
            )
        )
        if n_factors > 1
        else []
    )
    boundaries = [0, *cuts, total_ndim]
    dims = [boundaries[i + 1] - boundaries[i] for i in range(n_factors)]

    factors = tuple(
        draw(cast(Any, manifolds)(exclude=(cxm.CartesianProductManifold,), ndim=d))
        for d in dims
    )
    factor_names = tuple(f"f{i}" for i in range(n_factors))
    return cxm.CartesianProductManifold(factors=factors, factor_names=factor_names)
