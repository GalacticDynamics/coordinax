"""Hypothesis strategies for coordinax representations."""

__all__ = (
    "can_coord_map",
    "representation_classes",
    "representations",
    "representations_like",
    "representation_time_chain",
)


from typing import Any, Final
from typing_extensions import get_annotations

import hypothesis.strategies as st
import plum
from hypothesis import assume

import unxt as u

import coordinax as cx
from .utils import (
    Metadata,
    build_init_kwargs_strategy,
    draw_if_strategy,
    get_all_subclasses,
    strategy_for_annotation,
    wrap_if_not_inspectable,
)

# =============================================================================
# Main strategies


def can_coord_map(to_rep: cx.r.AbstractRep, from_rep: cx.r.AbstractRep, /) -> bool:
    """Return True if ``coord_map`` can convert between the two reps."""
    if type(to_rep) is type(from_rep):
        return True
    try:
        _ = to_rep.cartesian
        _ = from_rep.cartesian
    except (NotImplementedError, ValueError):
        return False
    return True


@st.composite  # type: ignore[untyped-decorator]
def representation_classes(
    draw: st.DrawFn,
    /,
    filter: type
    | tuple[type, ...]
    | st.SearchStrategy[type | tuple[type, ...]] = object,
    *,
    exclude_abstract: bool | st.SearchStrategy[bool] = True,
    exclude: tuple[type, ...] = (),
) -> type[cx.r.AbstractRep[Any, Any]]:
    """Strategy to draw representation classes.

    Parameters
    ----------
    draw
        Hypothesis draw function. Automatically provided by hypothesis.
    filter
        A class or tuple of classes to limit the representations to, by default
        `object`.  Can be a single type or a tuple of types.
    exclude_abstract
        Whether to exclude abstract subclasses, by default True.
    exclude
        Specific classes to exclude, by default
        ``(coordinax.r.Abstract0D, coordinax.r.TwoSphere)``.

    Returns
    -------
    type[AbstractRep]
        A representation class.

    Examples
    --------
    >>> from coordinax import r
    >>> import coordinax_hypothesis as cxst

    >>> # Draw any representation class
    >>> rep_class_strategy = cxst.representation_classes()
    >>> # Draw 1D velocity representation classes only
    >>> vel_1d_rep_class_strategy = cxst.representation_classes(
    ...     filter=r.Abstract1D)

    """
    classes = get_all_subclasses(
        cx.r.AbstractRep,
        filter=draw_if_strategy(draw, filter),
        exclude_abstract=draw_if_strategy(draw, exclude_abstract),
        exclude=exclude,
    )
    return draw(st.sampled_from(classes))


# ===================================================================


@st.composite  # type: ignore[untyped-decorator]
def representations(
    draw: st.DrawFn,
    /,
    filter: type | tuple[type, ...] | st.SearchStrategy[type | tuple[type, ...]] = (),
    *,
    exclude: tuple[type, ...] = (cx.r.Abstract0D, cx.r.TwoSphere),
    dimensionality: (int | None | st.SearchStrategy[int | None]) = None,
) -> cx.r.AbstractRep[Any, Any]:
    """Strategy to draw representation instances.

    Parameters
    ----------
    draw
        The draw function used by the hypothesis composite strategy.
        Automatically provided by hypothesis.
    filter
        A class or tuple of classes to limit the representations to, by default
        `()` (no additional filter). Can be a single type or a tuple of types.

        For example:
        - `coordinax.r.Abstract0D` to limit to 0D representations.
        - `coordinax.r.Abstract1D` to limit to 1D representations.
        - `coordinax.r.Abstract2D` to limit to 2D representations.
        - `coordinax.r.Abstract3D` to limit to 3D representations.

        In combination, this can be used to draw representations
        that satisfy multiple criteria, e.g.,
        `filter=(coordinax.r.Abstract3D, coordinax.r.AbstractSpherical3D)`.
    exclude
        Specific classes to exclude, by default ().
    dimensionality
        Dimensionality constraint for the representation. Can be:
        - `None`: No constraint
        - An integer: Exact dimensionality match
        - A strategy: Draw dimensionality from strategy (e.g.,
          `st.integers(min_value=1, max_value=2)`)

    Returns
    -------
    AbstractRep
        An instance of a representation class.

    Examples
    --------
    >>> from coordinax import r
    >>> import coordinax_hypothesis as cxst
    >>> import hypothesis.strategies as st

    >>> # Draw any representation instance (dimensionality > 0 by default)
    >>> rep_strategy = cxst.representations()
    >>> rep = rep_strategy.example()  # doctest: +SKIP
    >>> isinstance(rep, r.AbstractRep)  # doctest: +SKIP
    True
    >>> rep.dimensionality > 0  # doctest: +SKIP
    True

    >>> # Draw representations with exact dimensionality
    >>> exact_2d_strategy = cxst.representations(dimensionality=2)
    >>> exact_2d_rep = exact_2d_strategy.example()  # doctest: +SKIP
    >>> exact_2d_rep.dimensionality == 2  # doctest: +SKIP
    True

    >>> # Include 0-dimensional representations
    >>> all_dim_strategy = cxst.representations(dimensionality=None, exclude=())
    >>> all_dim_rep = all_dim_strategy.example()  # doctest: +SKIP
    >>> isinstance(all_dim_rep, r.AbstractRep)  # doctest: +SKIP
    True

    >>> # Use a strategy to draw dimensionality
    >>> strategy_dim = cxst.representations(
    ...     dimensionality=st.integers(min_value=1, max_value=2))
    >>> strategy_dim_rep = strategy_dim.example()  # doctest: +SKIP
    >>> 1 <= strategy_dim_rep.dimensionality <= 2  # doctest: +SKIP
    True

    """
    # Handle dimensionality parameter
    dimensionality = draw_if_strategy(draw, dimensionality)
    if isinstance(dimensionality, int) and dimensionality != 2:
        exclude = exclude + (cx.r.EmbeddedManifold,)
    # Exclude all dimensional flags except the target
    if isinstance(dimensionality, int) and dimensionality in cx.r.DIMENSIONAL_FLAGS:
        exclude = exclude + tuple(
            flag for i, flag in cx.r.DIMENSIONAL_FLAGS.items() if i != dimensionality
        )

    # Draw the representation class
    rep_cls = draw(
        representation_classes(
            filter=draw_if_strategy(draw, filter),
            exclude_abstract=True,
            exclude=exclude,
        )
    )

    # Build and draw kwargs for required parameters
    kwargs_strategy = build_init_kwargs_strategy(rep_cls, dim=dimensionality)
    kwargs = draw(kwargs_strategy)

    # Create the instance
    rep = rep_cls(**kwargs)

    # Filter by dimensionality if specified
    if dimensionality is not None:
        assume(rep.dimensionality == dimensionality)

    return rep


@plum.dispatch
def build_init_kwargs_strategy(
    cls: type[cx.r.SpaceTimeCT], /, *, dim: int | None
) -> st.SearchStrategy:
    """Specialized strategy for SpaceTimeCT classes.

    Parameters
    ----------
    cls : type[cx.r.SpaceTimeCT]
        The SpaceTimeCT class.
    dim : int | None
        The required dimensionality for the spatial_kind, or None for any dimensionality.

    Returns
    -------
    st.SearchStrategy[dict[str, Any]]
        A strategy that generates dictionaries with 'spatial_kind' key.
        The 'c' parameter is optional and uses the default value.

    """
    # Generate spatial_kind: any AbstractRep except SpaceTimeCT itself
    # If dimensionality is specified, use it; otherwise allow any dimensionality > 0
    spatial_kind_strategy = representations(
        exclude=(cx.r.SpaceTimeCT,),
        dimensionality=dim if dim is None else dim - 1,
    )
    # Generate 'c' parameter: either use default or draw from annotation
    c = st.one_of(
        st.just(cls.__dataclass_fields__["c"].default),
        strategy_for_annotation(
            wrap_if_not_inspectable(get_annotations(cls)["c"]), meta=Metadata()
        ),
    )
    return st.fixed_dictionaries({"spatial_kind": spatial_kind_strategy, "c": c})


@plum.dispatch
def build_init_kwargs_strategy(
    cls: type[cx.r.SpaceTimeEuclidean], /, *, dim: int | None
) -> st.SearchStrategy:
    """Specialized strategy for SpaceTimeEuclidean classes."""
    spatial_kind_strategy = representations(
        exclude=(cx.r.SpaceTimeEuclidean,),
        dimensionality=dim if dim is None else dim - 1,
    )
    c = st.one_of(
        st.just(cls.__dataclass_fields__["c"].default),
        strategy_for_annotation(
            wrap_if_not_inspectable(get_annotations(cls)["c"]), meta=Metadata()
        ),
    )
    return st.fixed_dictionaries({"spatial_kind": spatial_kind_strategy, "c": c})


@plum.dispatch
def build_init_kwargs_strategy(
    cls: type[cx.r.EmbeddedManifold], /, *, dim: int | None
) -> st.SearchStrategy:
    """Specialized strategy for EmbeddedManifold.

    Currently supports TwoSphere embedded in Cart3D with a length scale ``R``.
    """
    del cls, dim
    R = st.floats(
        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
    ).map(lambda v: u.Quantity(v, "km"))
    params = st.fixed_dictionaries({"R": R})
    return st.fixed_dictionaries(
        {
            "chart_kind": st.just(cx.r.twosphere),
            "ambient_kind": st.just(cx.r.cart3d),
            "params": params,
        }
    )


@st.composite  # type: ignore[untyped-decorator]
def representations_like(
    draw: st.DrawFn,
    /,
    representation: cx.r.AbstractRep[Any, Any]
    | st.SearchStrategy[cx.r.AbstractRep[Any, Any]],
) -> cx.r.AbstractRep[Any, Any]:
    """Generate representations similar to the provided one.

    This strategy inspects the provided representation to determine its flags
    (e.g., Abstract1D, Abstract2D, Abstract3D, AbstractSpherical3D, etc.) and
    dimensionality, then generates new representations matching those criteria.

    Parameters
    ----------
    draw
        The draw function used by the hypothesis composite strategy.
        Automatically provided by hypothesis.
    representation
        The template representation to match, or a strategy that generates one.

    Returns
    -------
    AbstractRep
        A new representation instance with the same flags and dimensionality
        as the template.

    Examples
    --------
    >>> from coordinax import r
    >>> import coordinax_hypothesis as cxst
    >>> from hypothesis import given

    >>> # Generate representations
    >>> @given(rep=cxst.representations_like(r.cart3d))
    ... def test_3d(rep):
    ...     assert isinstance(rep, r.Abstract3D)
    ...     assert rep.dimensionality == 3

    >>> # Generate representations like a 2D representation
    >>> @given(rep=cxst.representations_like(r.polar2d))
    ... def test_2d(rep):
    ...     assert isinstance(rep, r.Abstract2D)
    ...     assert rep.dimensionality == 2

    """
    # Draw the template representation if it's a strategy
    template = draw_if_strategy(draw, representation)

    # Extract flags by looking through the MRO for AbstractDimensionalFlag subclasses
    flags = tuple(
        base
        for base in type(template).__mro__
        if (
            isinstance(base, type)
            and issubclass(base, cx.r.AbstractDimensionalFlag)
            and base is not cx.r.AbstractDimensionalFlag
        )
    )

    # If no flags found, just use the base type
    if not flags:
        flags = (type(template),)

    # Generate a new representation with the same flags and dimensionality.
    # Keep the template's class available even if it's excluded by default.
    exclude = tuple(
        ex for ex in (cx.r.Abstract0D, cx.r.TwoSphere) if not isinstance(template, ex)
    )
    rep = draw(
        representations(
            filter=flags, dimensionality=template.dimensionality, exclude=exclude
        )
    )
    assume(can_coord_map(rep, template))
    assume(can_coord_map(template, rep))
    return rep


MAX_TIME_CHAIN_ITERS: Final = 50
MAX_TIME_CHAIN_ITER_MSG: Final = (
    f"Exceeded maximum iterations ({MAX_TIME_CHAIN_ITERS}) while building "
    "time antiderivative chain. This likely indicates a bug in the "
    "representation's time_antiderivative property."
)


@st.composite  # type: ignore[untyped-decorator]
def representation_time_chain(
    draw: st.DrawFn,
    role: type[cx.r.AbstractRoleFlag],
    rep: cx.r.AbstractRep[Any, Any] | st.SearchStrategy[cx.r.AbstractRep[Any, Any]],
    /,
) -> tuple[cx.r.AbstractRep[Any, Any], ...]:
    """Generate a chain of representations following time antiderivative pattern.

    Given a representation (position, velocity, or acceleration), this strategy
    returns a tuple containing representations that match the flags of each time
    antiderivative up to and including a position representation. Each element
    in the chain is generated using `representations_like()` to match the flags
    of the corresponding time antiderivative.

    Parameters
    ----------
    draw
        The draw function used by the hypothesis composite strategy.
        Automatically provided by hypothesis.
    role
        The role flag for the starting representation (e.g., `cx.r.Pos`,
        `cx.r.Vel`, or `cx.r.Acc`).
    rep
        The starting representation or a strategy that generates one.

    Returns
    -------
    tuple[AbstractRep, ...]
        A tuple of representations matching the time antiderivative chain.
        Each representation matches the flags of the corresponding time
        antiderivative but may be a different instance.
        - If input is position: (pos_rep,)
        - If input is velocity: (vel_rep, pos_rep)
        - If input is acceleration: (acc_rep, vel_rep, pos_rep)

    Examples
    --------
    >>> from coordinax import r
    >>> import coordinax_hypothesis as cxst

    >>> # Given an acceleration, get (acc, vel, pos) chain
    >>> @given(chain=cxst.representation_time_chain(r.Acc, r.cart3d))
    ... def test_chain(chain):
    ...     acc_rep, vel_rep, pos_rep = chain
    ...     assert isinstance(acc_rep, r.AbstractRep)
    ...     assert isinstance(vel_rep, r.AbstractRep)
    ...     assert isinstance(pos_rep, r.AbstractRep)

    """
    # Draw the starting representation if it's a strategy
    start_rep = draw_if_strategy(draw, rep)

    # Build the chain by following time_antiderivative until we reach position
    chain = [start_rep]
    current_role = role

    # Keep getting time antiderivatives until we reach a position representation
    # Safety: limit iterations to prevent infinite loops (no representation
    # should have more than a few time antiderivatives)
    i = 0
    while current_role is not cx.r.Pos:
        i += 1
        if i > MAX_TIME_CHAIN_ITERS:
            raise RuntimeError(MAX_TIME_CHAIN_ITER_MSG)

        # Get a representation like the time antiderivative
        current = draw(representations_like(representation=chain[-1]))

        # Store and move to next role
        chain.append(current)
        current_role = current_role.antiderivative()

    return tuple(chain)


# Register type strategy for Hypothesis's st.from_type()
# Note: Pass the callable, not an invoked strategy
st.register_type_strategy(cx.r.AbstractRep, lambda _: representations())

for flag_cls in get_all_subclasses(
    cx.r.AbstractDimensionalFlag, exclude_abstract=False
):
    # Skip representation base classes
    if issubclass(flag_cls, cx.r.AbstractRep):
        continue

    st.register_type_strategy(flag_cls, lambda typ: representations(typ))
