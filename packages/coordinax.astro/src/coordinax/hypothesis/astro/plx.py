"""Hypothesis strategies for Distance quantities."""

__all__ = ("parallaxes",)


from typing import Any, cast

from hypothesis import strategies as st

import unxt as u
import unxt_hypothesis as ust

import coordinax.astro as cxastro
from coordinax.hypothesis.distances._src.utils import make_nonnegative

ANGLE = u.dimension("angle")


@st.composite
def parallaxes(
    draw: st.DrawFn,
    /,
    *,
    check_negative: bool | st.SearchStrategy[bool] = True,
    **kwargs: Any,
) -> cxastro.Parallax:
    """Strategy for generating Parallax instances.

    Parameters
    ----------
    draw
        Hypothesis draw function. Automatically provided by hypothesis.
    check_negative
        Whether to enforce non-negative parallaxes. If `True` (default),
        generated parallaxes will be >= 0. Can be a hypothesis strategy to vary
        this behavior across test examples. Note that while theoretically
        parallax must be non-negative, noisy measurements can yield negative
        values.
    **kwargs
        Additional keyword arguments passed to `unxt_hypothesis.quantities`.
        Common options include 'dtype', 'shape', 'elements', 'unique'. The
        arguments 'unit' and 'quantity_cls' are set automatically and should not
        be provided.

    Returns
    -------
    coordinax.astro.Parallax
        A strategy that generates Parallax instances.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.hypothesis.astro as cxastrost
    >>> import coordinax.astro as cxastro

    >>> @given(plx=cxastrost.parallaxes())
    ... def test_parallax(plx):
    ...     assert isinstance(plx, cxastro.Parallax)
    ...     assert plx.value >= 0  # default check_negative=True

    With negative parallaxes allowed (for noisy measurements):

    >>> @given(plx=cxastrost.parallaxes(check_negative=False))
    ... def test_noisy_parallax(plx):
    ...     assert isinstance(plx, cxastro.Parallax)

    Generate parallax in specific units:

    >>> @given(plx=cxastrost.parallaxes(unit="mas"))
    ... def test_parallax_mas(plx):
    ...     assert plx.unit == "mas"

    """
    # Draw check_negative if it's a strategy
    check_negative = (
        draw(check_negative)
        if isinstance(check_negative, st.SearchStrategy)
        else check_negative
    )

    # Extract unit if provided (to avoid conflicts with dimension)
    # Default to angle dimension, but user can override with specific unit
    unit = kwargs.pop("unit", ANGLE)

    # Adjust elements strategy if needed to enforce non-negative values
    if check_negative:
        kwargs = make_nonnegative(draw, **kwargs)

    # Generate the Parallax quantity with angle dimension
    out = draw(
        ust.quantities(
            unit, quantity_cls=cxastro.Parallax, check_negative=check_negative, **kwargs
        )
    )
    return cast("cxastro.Parallax", out)


# Register type strategy for Hypothesis's st.from_type()
# Note: Pass the callable, not an invoked strategy
st.register_type_strategy(cxastro.Parallax, lambda _: parallaxes())  # ty: ignore[missing-argument]
