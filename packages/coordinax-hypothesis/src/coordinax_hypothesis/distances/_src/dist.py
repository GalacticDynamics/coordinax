"""Hypothesis strategies for Distance quantities."""

__all__ = ("distances",)


from typing import Any

from hypothesis import strategies as st

import unxt as u
import unxt_hypothesis as ust

import coordinax.distances as cxd
from .utils import make_nonnegative
from coordinax_hypothesis.utils import draw_if_strategy

LENGTH = u.dimension("length")


@st.composite
def distances(
    draw: st.DrawFn,
    /,
    *,
    check_negative: bool | st.SearchStrategy[bool] = True,
    **kwargs: Any,
) -> cxd.Distance:
    """Strategy for generating Distance instances.

    Parameters
    ----------
    draw
        Hypothesis draw function. Automatically provided by hypothesis.
    check_negative
        Whether to enforce non-negative distances. If `True` (default), generated
        distances will be >= 0. Can be a hypothesis strategy to vary this
        behavior across test examples.
    **kwargs
        Additional keyword arguments passed to `unxt_hypothesis.quantities`.
        Common options include 'dtype', 'shape', 'elements', 'unique'. The
        arguments 'unit' and 'quantity_cls' are set automatically and should not
        be provided.

    Returns
    -------
    coordinax.distance.Distance
        A strategy that generates Distance instances.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax_hypothesis.core as cxst
    >>> import coordinax.distances as cxd

    >>> @given(dist=cxst.distances())
    ... def test_distance(dist):
    ...     assert isinstance(dist, cxd.Distance)
    ...     assert dist.value >= 0  # default check_negative=True

    With negative distances allowed:

    >>> @given(dist=cxst.distances(check_negative=False))
    ... def test_signed_distance(dist):
    ...     assert isinstance(dist, cxd.Distance)

    """
    # Draw check_negative if it's a strategy
    check_negative = draw_if_strategy(draw, check_negative)

    # Extract unit if provided (to avoid conflicts with dimension)
    # Default to length dimension, but user can override with specific unit
    unit = kwargs.pop("unit", LENGTH)

    # Adjust elements strategy if needed to enforce non-negative values. This is
    # much more efficient than filtering out negative values after generation.
    if check_negative:
        kwargs = make_nonnegative(draw, **kwargs)

    # Generate the Distance quantity
    return draw(
        ust.quantities(
            unit,
            quantity_cls=cxd.Distance,
            check_negative=check_negative,
            **kwargs,
        )
    )


# Register type strategy for Hypothesis's st.from_type()
# Note: Pass the callable, not an invoked strategy
st.register_type_strategy(cxd.Distance, lambda _: distances())
