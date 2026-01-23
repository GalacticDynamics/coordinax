"""Hypothesis strategies for coordinax representations."""

__all__ = ("charts_like", "chart_time_chain")


from typing import Any, Final

import hypothesis.strategies as st
from hypothesis import assume

import coordinax as cx
import coordinax.charts as cxc
from .utils import can_point_transform
from coordinax_hypothesis._src.utils import draw_if_strategy


@st.composite  # type: ignore[untyped-decorator]
def charts_like(
    draw: st.DrawFn,
    /,
    representation: cxc.AbstractChart[Any, Any]
    | st.SearchStrategy[cxc.AbstractChart[Any, Any]],
) -> cxc.AbstractChart[Any, Any]:
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
    AbstractChart
        A new representation instance with the same flags and dimensionality
        as the template.

    Examples
    --------
    >>> import coordinax as cx
    >>> import coordinax_hypothesis as cxst
    >>> from hypothesis import given

    >>> # Generate representations
    >>> @given(chart=cxst.charts_like(r.cart3d))
    ... def test_3d(rep):
    ...     assert isinstance(rep, r.Abstract3D)
    ...     assert rep.ndim == 3

    >>> # Generate representations like a 2D representation
    >>> @given(chart=cxst.charts_like(r.polar2d))
    ... def test_2d(rep):
    ...     assert isinstance(rep, r.Abstract2D)
    ...     assert rep.ndim == 2

    """
    # Draw the template representation if it's a strategy
    template = draw_if_strategy(draw, representation)

    # Extract flags by looking through the MRO for AbstractDimensionalFlag subclasses
    flags = tuple(
        base
        for base in type(template).mro()
        if (
            issubclass(base, cxc.AbstractDimensionalFlag)
            and base is not cxc.AbstractDimensionalFlag
        )
    )

    # If no flags found, just use the base type
    if not flags:
        flags = (type(template),)

    # Generate a new representation with the same flags and dimensionality.
    # Keep the template's class available even if it's excluded by default.
    exclude = tuple(
        ex for ex in (cxc.Abstract0D, cxc.TwoSphere) if not isinstance(template, ex)
    )
    chart = draw(charts(filter=flags, dimensionality=template.ndim, exclude=exclude))
    assume(can_point_transform(chart, template))
    assume(can_point_transform(template, chart))
    return chart


MAX_TIME_CHAIN_ITERS: Final = 50
MAX_TIME_CHAIN_ITER_MSG: Final = (
    f"Exceeded maximum iterations ({MAX_TIME_CHAIN_ITERS}) while building "
    "time antiderivative chain. This likely indicates a bug in the "
    "representation's time_antiderivative property."
)


@st.composite  # type: ignore[untyped-decorator]
def chart_time_chain(
    draw: st.DrawFn,
    role_cls: type[cx.roles.AbstractRole],
    chart: cxc.AbstractChart[Any, Any] | st.SearchStrategy[cxc.AbstractChart[Any, Any]],
    /,
) -> tuple[cxc.AbstractChart[Any, Any], ...]:
    """Generate a chain of representations following time antiderivative pattern.

    Given a representation (position, velocity, or acceleration), this strategy
    returns a tuple containing representations that match the flags of each time
    antiderivative up to and including a position representation. Each element
    in the chain is generated using `charts_like()` to match the flags
    of the corresponding time antiderivative.

    Parameters
    ----------
    draw
        The draw function used by the hypothesis composite strategy.
        Automatically provided by hypothesis.
    role_cls
        The role flag for the starting representation (e.g., `cx.roles.PhysDisp`,
        `cx.roles.PhysVel`, or `cx.roles.PhysAcc`).
    chart
        The starting chart or a strategy that generates one.

    Returns
    -------
    tuple[AbstractChart, ...]
        A tuple of representations matching the time antiderivative chain.
        Each representation matches the flags of the corresponding time
        antiderivative but may be a different instance.
        - If input is position: (pos_rep,)
        - If input is velocity: (vel_rep, pos_rep)
        - If input is acceleration: (acc_rep, vel_rep, pos_rep)

    Examples
    --------
    >>> import coordinax as cx
    >>> import coordinax_hypothesis as cxst

    >>> # Given an acceleration, get (acc, vel, pos) chain
    >>> @given(chain=cxst.chart_time_chain(cx.roles.PhysAcc, cxc.cart3d))
    ... def test_chain(chain):
    ...     acc_rep, vel_rep, pos_rep = chain
    ...     assert isinstance(acc_rep, cxc.AbstractChart)
    ...     assert isinstance(vel_rep, cxc.AbstractChart)
    ...     assert isinstance(pos_rep, cxc.AbstractChart)

    """
    # Draw the starting representation if it's a strategy
    start_chart = draw_if_strategy(draw, chart)

    # Build the chain by following time_antiderivative until we reach position
    chain = [start_chart]
    current_role = role_cls()

    # Keep getting time antiderivatives until we reach a position representation
    # Safety: limit iterations to prevent infinite loops (no representation
    # should have more than a few time antiderivatives)
    i = 0
    while not isinstance(current_role, cx.roles.PhysDisp):
        i += 1
        if i > MAX_TIME_CHAIN_ITERS:
            raise RuntimeError(MAX_TIME_CHAIN_ITER_MSG)

        # Get a representation like the time antiderivative
        current = draw(charts_like(representation=chain[-1]))

        # Store and move to next role
        chain.append(current)
        current_role = current_role.antiderivative()

    return tuple(chain)
