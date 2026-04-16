"""Hypothesis strategies for coordinax atlases."""

__all__ = ("atlas_classes", "atlases")

import inspect
from typing import Any, cast

import hypothesis.strategies as st
import plum
from hypothesis import assume

import coordinax.charts as cxc
import coordinax.manifolds as cxm

from coordinax.hypothesis.utils import (
    draw_if_strategy,
    get_all_subclasses,
    strip_return_annotation,
)


@st.composite
def atlas_classes(
    draw: st.DrawFn,
    /,
    *,
    filter: type
    | tuple[type, ...]
    | st.SearchStrategy[type | tuple[type, ...]] = object,
    exclude_abstract: bool | st.SearchStrategy[bool] = True,
    exclude: tuple[type, ...] = (),
) -> type[Any]:
    """Strategy to draw atlas classes (not instances) from the atlas hierarchy.

    Parameters
    ----------
    draw
        Hypothesis draw function. Automatically provided by hypothesis.
    filter
        A class or tuple of classes to restrict the drawn atlas classes to, by
        default `object` (no restriction). Only subclasses of `AbstractAtlas`
        that are also subclasses of the given type(s) will be considered.

        For example:

        - `coordinax.manifolds.EuclideanAtlas` to draw only Euclidean atlases.
        - `coordinax.manifolds.CustomAtlas` to draw only custom atlases.
        - `(coordinax.manifolds.EuclideanAtlas,
          coordinax.manifolds.HyperSphericalAtlas)` to draw either type.

    exclude_abstract
        Whether to exclude abstract atlas subclasses, by default `True`.
        Set to `False` to include abstract base classes in the drawn classes.
    exclude
        Specific atlas classes to exclude from the drawn classes, by default
        `()`.

    Returns
    -------
    type[AbstractAtlas]
        An atlas class (not an instance).

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.hypothesis.manifolds as cxmst

    >>> # Draw any concrete atlas class
    >>> atlas_class_strategy = cxmst.atlas_classes()

    >>> # Draw only EuclideanAtlas classes
    >>> euclidean_strategy = cxmst.atlas_classes(filter=cxm.EuclideanAtlas)

    >>> # Exclude CartesianProductAtlas from the draw
    >>> no_product_strategy = cxmst.atlas_classes(
    ...     exclude=(cxm.CartesianProductAtlas,)
    ... )

    """
    classes = get_all_subclasses(
        cxm.AbstractAtlas,
        filter=draw_if_strategy(draw, filter),
        exclude_abstract=draw_if_strategy(draw, exclude_abstract),
        exclude=exclude,
    )
    return cast(type[Any], draw(st.sampled_from(classes)))


#####################################################################


@plum.dispatch.abstract
def atlases(
    draw: st.DrawFn,
    atlas_cls: Any,
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = object,
    exclude: tuple[type, ...] = (),
    ndim: int | st.SearchStrategy[int] | None = None,
    required_chart_classes: tuple[type[cxc.AbstractChart], ...] = (),
) -> Any:
    """Strategy to draw atlas instances across the concrete atlas hierarchy.

    This is a multiple-dispatch strategy. The behaviour depends on the type of
    the first positional argument ``atlas_cls``:

    - **No argument** (``atlas_cls=None``): draws any concrete atlas class,
      filtered by ``filter``, ``exclude``, ``ndim``, and
      ``required_chart_classes``, then delegates to the class-specific
      dispatch.
    - **A** ``SearchStrategy``: draws an atlas class from the strategy, then
      delegates to the class-specific dispatch.
    - **An abstract** ``AbstractAtlas`` **subclass**: treats the class as a
      filter and redispatches without a class argument.
    - **``EuclideanAtlas``**: returns a ``EuclideanAtlas(dim)`` with ``dim``
      in ``[0, 3]``, constrained by ``ndim`` when given.
    - **``HyperSphericalAtlas``**: returns a ``HyperSphericalAtlas()``. Only compatible
      with ``ndim=2``; hypothesis discards the example otherwise.
    - **``CustomAtlas``**: returns a ``CustomAtlas`` whose charts are drawn
      from the concrete ``AbstractChart`` subclasses with the requested
      dimensionality. The ``required_chart_classes`` argument pins specific
      chart types into the atlas.
    - **``CartesianProductAtlas``**: returns a ``CartesianProductAtlas``
      built from two factor atlases whose dimensionalities sum to ``ndim``
      (at least 2). ``CartesianProductAtlas`` is excluded from the factors.

    Parameters
    ----------
    draw
        Hypothesis draw function. Automatically provided by hypothesis.
    atlas_cls
        Selects the dispatch path:

        - ``None`` (default): draw from all concrete atlas types.
        - A ``SearchStrategy``: draw a class from the strategy.
        - An **abstract** subclass of ``AbstractAtlas``: filter to that
          subtree.
        - A **concrete** subclass of ``AbstractAtlas``: draw an instance of
          exactly that class (``EuclideanAtlas``, ``HyperSphericalAtlas``,
          ``CustomAtlas``, ``CartesianProductAtlas``).

    filter
        Restrict the pool of atlas classes to those that are subclasses of
        the given type(s). Only used when ``atlas_cls`` is ``None``. Ignored
        (and must be empty) when ``atlas_cls`` is provided.
    exclude
        Specific atlas classes to exclude from the candidate pool. Only used
        when ``atlas_cls`` is ``None``. Ignored (and must be empty) when
        ``atlas_cls`` is provided.
    ndim
        Dimensionality constraint on the drawn atlas. May be:

        - ``None``: no constraint (default).
        - An integer: the atlas must support exactly this dimensionality.
        - A ``SearchStrategy[int]``: draw the dimensionality from the
          strategy.

        Not all atlas types support all dimensionalities; incompatible
        examples are discarded via ``hypothesis.assume``.
    required_chart_classes
        A tuple of ``AbstractChart`` subclasses that **must** appear in the
        drawn ``CustomAtlas``. Only meaningful when drawing ``CustomAtlas``
        instances; raises ``ValueError`` for other concrete atlas types.
        Each class must be zero-argument constructible, and all must share
        the same ``ndim`` as the drawn atlas.

    Returns
    -------
    AbstractAtlas
        An atlas instance.

    Raises
    ------
    ValueError
        If ``filter`` or ``exclude`` are non-empty when ``atlas_cls`` is a
        concrete class or a ``SearchStrategy``.
    ValueError
        If ``ndim`` is not ``None`` when ``atlas_cls`` is a
        ``SearchStrategy``.
    ValueError
        If ``required_chart_classes`` is non-empty for any concrete atlas
        type other than ``CustomAtlas``.
    ValueError
        If a class in ``required_chart_classes`` is not zero-argument
        constructible, or its ``ndim`` does not match the requested
        dimensionality.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.hypothesis.manifolds as cxmst
    >>> import hypothesis.strategies as st

    Draw any concrete atlas (no argument):

    >>> any_atlas = cxmst.atlases()

    Constrain to a specific dimensionality:

    >>> atlas_2d = cxmst.atlases(ndim=2)
    >>> atlas_3d = cxmst.atlases(ndim=3)

    Draw the dimensionality from a strategy:

    >>> atlas_1or2d = cxmst.atlases(ndim=st.integers(min_value=1, max_value=2))

    Filter to a subset of atlas types:

    >>> euclidean_or_sphere = cxmst.atlases(
    ...     filter=(cxm.EuclideanAtlas, cxm.HyperSphericalAtlas)
    ... )

    Exclude a specific atlas type:

    >>> no_product = cxmst.atlases(exclude=(cxm.CartesianProductAtlas,))

    Pass a concrete class to draw an instance of exactly that type:

    >>> euclidean = cxmst.atlases(cxm.EuclideanAtlas)
    >>> euclidean_2d = cxmst.atlases(cxm.EuclideanAtlas, ndim=2)
    >>> sphere = cxmst.atlases(cxm.HyperSphericalAtlas)
    >>> custom = cxmst.atlases(cxm.CustomAtlas)
    >>> product = cxmst.atlases(cxm.CartesianProductAtlas)
    >>> product_4d = cxmst.atlases(cxm.CartesianProductAtlas, ndim=4)

    Require specific chart classes in a ``CustomAtlas``:

    >>> custom_with_cart = cxmst.atlases(
    ...     cxm.CustomAtlas,
    ...     required_chart_classes=(cxc.Cart3D,),
    ...     ndim=3
    ... )

    Pass an abstract class to filter to that subtree:

    >>> abstract_filter = cxmst.atlases(cxm.AbstractAtlas)

    Pass a strategy of atlas classes:

    >>> from_strategy = cxmst.atlases(
    ...     st.sampled_from([cxm.EuclideanAtlas, cxm.HyperSphericalAtlas])
    ... )

    """
    raise NotImplementedError  # pragma: no cover


# ---------------------------------------------------------------------------
# ndim-compatibility helper — extend by adding a new dispatch for each new
# concrete atlas type.
# ---------------------------------------------------------------------------


@plum.dispatch
def _atlas_class_supports_ndim(
    cls: type[cxm.EuclideanAtlas],
    ndim: int,
    /,
) -> bool:
    """EuclideanAtlas supports dimensions 0-3."""
    return ndim <= 3


@plum.dispatch
def _atlas_class_supports_ndim(
    cls: type[cxm.HyperSphericalAtlas],
    ndim: int,
    /,
) -> bool:
    """HyperSphericalAtlas is always 2-D."""
    return ndim == 2


@plum.dispatch
def _atlas_class_supports_ndim(
    cls: type[cxm.CartesianProductAtlas],
    ndim: int,
    /,
) -> bool:
    """CartesianProductAtlas supports any positive dimensionality (≥1 factor)."""
    return ndim >= 1


@plum.dispatch
def _atlas_class_supports_ndim(
    cls: type[cxm.CustomAtlas],
    ndim: int,
    /,
) -> bool:
    """CustomAtlas supports any positive dimensionality."""
    return ndim >= 1


@plum.dispatch
def _atlas_class_supports_ndim(
    cls: type[cxm.AbstractAtlas],
    ndim: int,
    /,
) -> bool:
    """Fallback: unknown atlas types are assumed to support any ndim."""
    return True


# ---------------------------------------------------------------------------


@plum.dispatch
@strip_return_annotation
@st.composite
def atlases(  # noqa: F811
    draw: st.DrawFn,
    atlas_cls: None = None,
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = object,
    exclude: tuple[type, ...] = (),
    ndim: int | st.SearchStrategy[int] | None = None,
    required_chart_classes: tuple[type[cxc.AbstractChart], ...] = (),
) -> cxm.AbstractAtlas:
    """Draw any concrete atlas, filtered by ``filter``, ``exclude``, and ``ndim``.

    Selects a concrete atlas class from the eligible pool and redispatches to
    the class-specific strategy.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.hypothesis.manifolds as cxmst
    >>> import hypothesis.strategies as st

    >>> # Draw any concrete atlas
    >>> any_atlas = cxmst.atlases()

    >>> # Limit to atlases compatible with ndim=2
    >>> atlas_2d = cxmst.atlases(ndim=2)

    >>> # Exclude CartesianProductAtlas from the draw
    >>> no_product = cxmst.atlases(exclude=(cxm.CartesianProductAtlas,))

    >>> # Filter to EuclideanAtlas and HyperSphericalAtlas only
    >>> subset = cxmst.atlases(filter=(cxm.EuclideanAtlas, cxm.HyperSphericalAtlas))

    """
    # Process inputs
    target_ndim = draw_if_strategy(draw, ndim)
    chosen_filter = draw_if_strategy(draw, filter)

    all_classes = get_all_subclasses(
        cxm.AbstractAtlas,
        filter=chosen_filter,
        exclude_abstract=True,
        exclude=exclude,
    )
    classes = tuple(
        cls
        for cls in all_classes
        if (target_ndim is None or _atlas_class_supports_ndim(cls, target_ndim))
        and (not required_chart_classes or issubclass(cls, cxm.CustomAtlas))
    )
    if not classes:
        assume(False)

    # Draw and redispatch
    selected_cls = cast(type[cxm.AbstractAtlas], draw(st.sampled_from(classes)))
    kwargs: dict[str, Any] = {"ndim": target_ndim}
    if issubclass(selected_cls, cxm.CustomAtlas):
        kwargs["required_chart_classes"] = required_chart_classes
    return draw(cast(Any, atlases)(selected_cls, **kwargs))


@plum.dispatch
@strip_return_annotation
@st.composite
def atlases(  # noqa: F811
    draw: st.DrawFn,
    atlas_cls: st.SearchStrategy,
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: Any = None,
    required_chart_classes: tuple[type[cxc.AbstractChart], ...] = (),
) -> Any:
    """Draw an atlas class from a strategy, then redispatch.

    The drawn class is passed to the class-specific dispatch. ``filter``,
    ``exclude``, and ``ndim`` must not be used alongside a strategy argument.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.hypothesis.manifolds as cxmst
    >>> import hypothesis.strategies as st

    >>> # Draw a class from an explicit strategy, then draw an instance
    >>> from_strategy = cxmst.atlases(
    ...     st.sampled_from([cxm.EuclideanAtlas, cxm.HyperSphericalAtlas])
    ... )

    """
    # Check input
    if filter or exclude:
        raise ValueError(
            "When atlas_cls is provided, filter and exclude must be empty."
        )
    if ndim is not None:
        raise ValueError("When atlas_cls is provided, ndim must be None.")

    # Draw and redispatch
    selected_cls = draw(atlas_cls)
    return draw(
        cast(Any, atlases)(
            selected_cls,
            ndim=ndim,
            required_chart_classes=required_chart_classes,
        )
    )


@plum.dispatch
@strip_return_annotation
@st.composite
def atlases(  # noqa: F811
    draw: st.DrawFn,
    atlas_cls: type[cxm.AbstractAtlas],
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: int | st.SearchStrategy | None = None,
    required_chart_classes: tuple[type[cxc.AbstractChart], ...] = (),
) -> Any:
    """Draw atlas instances from abstract or concrete class selectors.

    For abstract classes, the class is treated as a ``filter`` and the call
    redispatches to the no-argument dispatch. For concrete classes, the call
    redispatches directly to the most specific concrete-class overload.
    ``filter`` and ``exclude`` must be empty.

    Examples
    --------
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.hypothesis.manifolds as cxmst

    >>> # Draw any concrete subclass of AbstractAtlas
    >>> from_abstract = cxmst.atlases(cxm.AbstractAtlas)

    """
    # Check input
    if filter or exclude:
        msg = "When atlas_cls is provided, filter and exclude must be empty."
        raise ValueError(msg)

    if not inspect.isabstract(atlas_cls):
        kwargs: dict[str, Any] = {"ndim": ndim}
        if issubclass(atlas_cls, cxm.CustomAtlas):
            kwargs["required_chart_classes"] = required_chart_classes
        return draw(cast(Any, atlases)(atlas_cls, **kwargs))

    # Draw and redispatch
    return draw(
        cast(Any, atlases)(
            filter=atlas_cls,
            exclude=(),
            ndim=ndim,
            required_chart_classes=required_chart_classes,
        )
    )


@plum.dispatch
@strip_return_annotation
@st.composite
def atlases(  # noqa: F811
    draw: st.DrawFn,
    atlas_cls: type[cxm.EuclideanAtlas],
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: int | st.SearchStrategy | None = None,
) -> Any:
    """Draw a ``EuclideanAtlas`` with dimensionality in ``[0, 3]``.

    If ``ndim`` is given, the atlas is constructed with that exact dimension
    (clamped to ``[0, 3]``). Otherwise, the dimension is drawn uniformly.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.hypothesis.manifolds as cxmst

    >>> # Draw a EuclideanAtlas of any dimension in [0, 3]
    >>> euclidean = cxmst.atlases(cxm.EuclideanAtlas)

    >>> # Draw a 3-D EuclideanAtlas
    >>> euclidean_3d = cxmst.atlases(cxm.EuclideanAtlas, ndim=3)

    """
    # Examine `ndim`
    target_ndim = draw_if_strategy(draw, ndim)

    # TODO: enable higher dimensions
    if target_ndim is None:
        dim = draw(st.integers(min_value=0, max_value=3))
    else:
        dim = max(0, min(target_ndim, 3))

    # Construct
    return cxm.EuclideanAtlas(dim)


@plum.dispatch
@strip_return_annotation
@st.composite
def atlases(  # noqa: F811
    draw: st.DrawFn,
    atlas_cls: type[cxm.HyperSphericalAtlas],
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: int | st.SearchStrategy | None = None,
) -> Any:
    """Draw a ``HyperSphericalAtlas`` (always 2-D).

    Examples with ``ndim != 2`` are discarded via ``hypothesis.assume``.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.hypothesis.manifolds as cxmst

    >>> sphere = cxmst.atlases(cxm.HyperSphericalAtlas)

    """
    target_ndim = draw_if_strategy(draw, ndim)
    if target_ndim is not None and target_ndim != 2:
        assume(False)
    return cxm.HyperSphericalAtlas()


@plum.dispatch
@strip_return_annotation
@st.composite
def atlases(  # noqa: F811
    draw: st.DrawFn,
    atlas_cls: type[cxm.CustomAtlas],
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: int | st.SearchStrategy | None = None,
    required_chart_classes: tuple[type[cxc.AbstractChart], ...] = (),
) -> Any:
    """Draw a ``CustomAtlas`` with charts drawn from the concrete chart hierarchy.

    The atlas dimensionality is drawn from ``[1, 3]`` unless ``ndim`` pins it.
    ``required_chart_classes`` guarantees specific chart types are included.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.hypothesis.manifolds as cxmst

    >>> # Draw any CustomAtlas
    >>> custom = cxmst.atlases(cxm.CustomAtlas)

    >>> # Draw a 3-D CustomAtlas
    >>> custom_3d = cxmst.atlases(cxm.CustomAtlas, ndim=3)

    >>> # Require Cartesian3D to appear in the drawn atlas
    >>> custom_with_cart = cxmst.atlases(
    ...     cxm.CustomAtlas,
    ...     required_chart_classes=(cxc.Cart3D,),
    ...     ndim=3,
    ... )

    """
    target_ndim = draw_if_strategy(draw, ndim)
    custom_ndim = (
        draw(st.integers(min_value=1, max_value=3))
        if target_ndim is None
        else target_ndim
    )

    for cls in required_chart_classes:
        try:
            chart = cls()
        except TypeError as exc:
            raise ValueError(
                f"Required chart class {cls.__name__} must be zero-argument constructible."
            ) from exc
        if chart.ndim != custom_ndim:
            raise ValueError(
                f"Required chart class {cls.__name__} has ndim={chart.ndim}, "
                f"expected ndim={custom_ndim}."
            )

    filtered_classes: list[type[cxc.AbstractChart[Any, Any]]] = []
    for cls in get_all_subclasses(cxc.AbstractChart, exclude_abstract=True):
        cls = cast(type[cxc.AbstractChart[Any, Any]], cls)
        try:
            chart = cls()
        except TypeError:
            continue
        if chart.ndim == custom_ndim:
            filtered_classes.append(cls)

    if not filtered_classes:
        assume(False)

    sampled_classes = draw(
        st.lists(
            st.sampled_from(tuple(filtered_classes)),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    classes = tuple(dict.fromkeys((*sampled_classes, *required_chart_classes)))
    default_cls = draw(st.sampled_from(classes))
    return cxm.CustomAtlas(charts=classes, chart_default=default_cls())


@plum.dispatch
@strip_return_annotation
@st.composite
def atlases(  # noqa: F811
    draw: st.DrawFn,
    atlas_cls: type[cxm.CartesianProductAtlas],
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: int | st.SearchStrategy | None = None,
) -> Any:
    """Draw a ``CartesianProductAtlas`` with 1-5 non-product factor atlases.

    The number of factors is drawn uniformly from 1 to 5. Each factor atlas is
    drawn as a non-product atlas with ndim in ``[1, 3]``, and the total
    dimensionality of the product equals the sum of those factor dimensionalities.
    When ``ndim`` is given, examples are constrained so a valid partition into
    ``n_factors`` values in ``[1, 3]`` exists.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.hypothesis.manifolds as cxmst

    >>> # Draw any CartesianProductAtlas (1-5 factors)
    >>> product = cxmst.atlases(cxm.CartesianProductAtlas)

    >>> # Draw a CartesianProductAtlas with total ndim=4
    >>> product_4d = cxmst.atlases(cxm.CartesianProductAtlas, ndim=4)

    """
    target_ndim = draw_if_strategy(draw, ndim)

    # Draw the number of factors: 1–5
    n_factors = draw(st.integers(min_value=1, max_value=5))

    dims: list[int] = []
    if target_ndim is None:
        dims = [draw(st.integers(min_value=1, max_value=3)) for _ in range(n_factors)]
    else:
        # Feasibility for partition of target_ndim into n_factors entries in [1, 3].
        assume(n_factors <= target_ndim <= 3 * n_factors)

        remaining = target_ndim
        for i in range(n_factors):
            factors_left_after = n_factors - i - 1
            min_this = max(1, remaining - 3 * factors_left_after)
            max_this = min(3, remaining - factors_left_after)
            dim_i = draw(st.integers(min_value=min_this, max_value=max_this))
            dims.append(dim_i)
            remaining -= dim_i

        assume(remaining == 0)

    factors = tuple(
        draw(cast(Any, atlases)(exclude=(cxm.CartesianProductAtlas,), ndim=d))
        for d in dims
    )
    factor_names = tuple(f"f{i}" for i in range(n_factors))
    return cxm.CartesianProductAtlas(factors=factors, factor_names=factor_names)
