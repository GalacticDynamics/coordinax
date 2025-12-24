"""Hypothesis strategies for Angle quantities."""

__all__ = ("angles",)

from typing import Any, TypeAlias

import unxt_hypothesis as ust
from hypothesis import strategies as st

import unxt as u

import coordinax as cx

WrapToArgs: TypeAlias = (
    tuple[
        u.Quantity | st.SearchStrategy[u.Quantity],
        u.Quantity | st.SearchStrategy[u.Quantity],
    ]
    | None
)


@st.composite  # type: ignore[untyped-decorator]
def angles(
    draw: st.DrawFn,
    *,
    wrap_to: WrapToArgs | st.SearchStrategy[WrapToArgs] = None,
    **kwargs: Any,
) -> u.Angle:
    """Strategy for generating Angle instances.

    Parameters
    ----------
    draw
        Hypothesis draw function. Automatically provided by hypothesis.
    wrap_to
        Strategy for generating (min, max) bounds for angle wrapping.
        If `None`, the angle will have no wrapping specified.
    **kwargs
        Additional keyword arguments passed to `unxt_hypothesis.quantities`.
        Common options include 'dtype', 'shape', 'elements', 'unique'.  The
        arguments 'unit' and 'quantity_cls' are set automatically and should not
        be provided.

    Returns
    -------
    coordinax.Angle
        A strategy that generates Angle instances.

    Examples
    --------
    >>> from hypothesis import given
    >>> from coordinax_hypothesis import angles
    >>> import unxt as u

    >>> @given(angle=angles())
    ... def test_angle(angle):
    ...     assert isinstance(angle, u.Angle)

    With wrapping:

    >>> from hypothesis import strategies as st
    >>> wrap_bounds = st.just((u.Q(0, "deg"), u.Q(360, "deg")))
    >>> @given(angle=angles(wrap_to=wrap_bounds))
    ... def test_wrapped_angle(angle):
    ...     assert angle.wrap_to is not None

    """
    # Determine wrapping bounds if provided
    if isinstance(wrap_to, st.SearchStrategy):
        wrap_to = draw(wrap_to)

    # Extract unit if provided (to avoid conflicts with dimension) Default to
    # angle dimension, but user can override with specific unit. If user
    # provides a bad unit, unxt_hypothesis.quantities will raise an error given
    # the `cx.angle.Angle` quantity_cls.
    unit = kwargs.pop("unit", u.dimension("angle"))

    # Generate the Angle quantity
    angle_strategy = ust.quantities(unit, quantity_cls=cx.angle.Angle, **kwargs)

    # Apply wrapping if bounds are provided
    if wrap_to is not None:
        wrap_min, wrap_max = wrap_to
        return draw(ust.wrap_to(angle_strategy, min=wrap_min, max=wrap_max))

    return draw(angle_strategy)
