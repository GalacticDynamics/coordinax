"""Hypothesis strategies for Distance quantities."""

__all__ = ("distances", "distance_moduli", "parallaxes")

import warnings

from collections.abc import Mapping
from typing import Any, assert_never

import jax.numpy as jnp
from hypothesis import strategies as st
from hypothesis.extra.array_api import make_strategies_namespace

import unxt as u
import unxt_hypothesis as ust

import coordinax.distance as cxd

xps = make_strategies_namespace(jnp)


def make_nonnegative(draw: st.DrawFn, /, **kwargs: Any) -> dict[str, Any]:
    """Adjust kwargs to ensure non-negative values in generated quantities.

    This helper function modifies the `elements` parameter in kwargs to ensure
    that generated values are non-negative. It handles three cases:

    1. If `elements` is a Mapping (dict), updates `min_value` to be at least 0
    2. If `elements` is a SearchStrategy, applies absolute value mapping
    3. If `elements` is not provided, creates a default strategy with min_value=0

    Parameters
    ----------
    draw
        Hypothesis draw function for drawing from strategies.
    **kwargs
        Keyword arguments that will be passed to `unxt_hypothesis.quantities`.
        The `elements` parameter will be modified if present, or created if absent.

    Returns
    -------
    dict[str, Any]
        Modified kwargs with adjusted `elements` parameter to ensure non-negative
        values.

    Examples
    --------
    >>> from hypothesis import strategies as st
    >>> import jax.numpy as jnp
    >>> # With custom elements
    >>> kwargs = {"elements": {"min_value": -10, "max_value": 100}}
    >>> # modified = make_nonnegative(draw, **kwargs)
    >>> # modified["elements"]["min_value"] will be 0

    >>> # Without elements, creates default non-negative strategy
    >>> kwargs = {"dtype": jnp.float32}
    >>> # modified = make_nonnegative(draw, **kwargs)
    >>> # modified will have "elements" key with min_value=0

    """
    if "elements" in kwargs:
        # User provided elements strategy - need to check if it's a float strategy
        # and adjust min_value if necessary
        elements: Mapping[str, Any] | st.SearchStrategy = kwargs["elements"]
        if isinstance(elements, Mapping):
            elements = dict(elements)
            elements["min_value"] = max(0.0, elements.get("min_value", 0.0))
        elif isinstance(elements, st.SearchStrategy):
            elements = elements.map(abs)  # Simple way to ensure non-negative
        else:
            assert_never(elements)
    else:
        # No elements provided, we need to set a default with min_value=0
        # Get dtype if specified, otherwise use default
        dtype = kwargs.get("dtype")
        if dtype is not None:
            dtype = draw(dtype) if isinstance(dtype, st.SearchStrategy) else dtype
        else:
            dtype = jnp.float32

        # Create elements strategy with min_value=0
        kwargs["elements"] = xps.from_dtype(
            dtype,
            min_value=0.0,
            # TODO: other kwargs?
        )
    return kwargs


@st.composite  # type: ignore[untyped-decorator]
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
    >>> from coordinax_hypothesis import distances
    >>> import coordinax.distance as cxd

    >>> @given(dist=distances())
    ... def test_distance(dist):
    ...     assert isinstance(dist, cxd.Distance)
    ...     assert dist.value >= 0  # default check_negative=True

    With negative distances allowed:

    >>> @given(dist=distances(check_negative=False))
    ... def test_signed_distance(dist):
    ...     assert isinstance(dist, cxd.Distance)

    """
    # Draw check_negative if it's a strategy
    check_negative = (
        draw(check_negative)
        if isinstance(check_negative, st.SearchStrategy)
        else check_negative
    )

    # Extract unit if provided (to avoid conflicts with dimension)
    # Default to length dimension, but user can override with specific unit
    unit = kwargs.pop("unit", u.dimension("length"))

    # Adjust elements strategy if needed to enforce non-negative values
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


@st.composite  # type: ignore[untyped-decorator]
def distance_moduli(
    draw: st.DrawFn,
    /,
    **kwargs: Any,
) -> cxd.DistanceModulus:
    """Strategy for generating DistanceModulus instances.

    Parameters
    ----------
    draw
        Hypothesis draw function. Automatically provided by hypothesis.
    **kwargs
        Additional keyword arguments passed to `unxt_hypothesis.quantities`.
        Common options include 'dtype', 'shape', 'elements', 'unique'. The
        arguments 'unit' and 'quantity_cls' are set automatically and should not
        be provided. Note that DistanceModulus always has units of 'mag'.

    Returns
    -------
    coordinax.distance.DistanceModulus
        A strategy that generates DistanceModulus instances.

    Warns
    -----
    UserWarning
        If 'unit' is specified in kwargs, since DistanceModulus always uses
        'mag' units and the argument will be ignored.

    Examples
    --------
    >>> from hypothesis import given
    >>> from coordinax_hypothesis import distance_moduli
    >>> import coordinax.distance as cxd

    >>> @given(dm=distance_moduli())
    ... def test_distance_modulus(dm):
    ...     assert isinstance(dm, cxd.DistanceModulus)
    ...     assert dm.unit == "mag"

    Generate distance modulus arrays:

    >>> @given(dm=distance_moduli(shape=10))
    ... def test_dm_array(dm):
    ...     assert dm.shape == (10,)

    """
    # DistanceModulus always uses 'mag' units
    # Don't allow user to override unit
    if "unit" in kwargs:
        warnings.warn(
            "DistanceModulus always uses 'mag' units. The 'unit' argument is ignored.",
            UserWarning,
            stacklevel=2,
        )
    kwargs.pop("unit", None)

    # Generate the DistanceModulus quantity
    return draw(
        ust.quantities(
            unit="mag",
            quantity_cls=cxd.DistanceModulus,
            **kwargs,
        )
    )


@st.composite  # type: ignore[untyped-decorator]
def parallaxes(
    draw: st.DrawFn,
    /,
    *,
    check_negative: bool | st.SearchStrategy[bool] = True,
    **kwargs: Any,
) -> cxd.Parallax:
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
    coordinax.distance.Parallax
        A strategy that generates Parallax instances.

    Examples
    --------
    >>> from hypothesis import given
    >>> from coordinax_hypothesis import parallaxes
    >>> import coordinax.distance as cxd

    >>> @given(plx=parallaxes())
    ... def test_parallax(plx):
    ...     assert isinstance(plx, cxd.Parallax)
    ...     assert plx.value >= 0  # default check_negative=True

    With negative parallaxes allowed (for noisy measurements):

    >>> @given(plx=parallaxes(check_negative=False))
    ... def test_noisy_parallax(plx):
    ...     assert isinstance(plx, cxd.Parallax)

    Generate parallax in specific units:

    >>> @given(plx=parallaxes(unit="mas"))
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
    unit = kwargs.pop("unit", u.dimension("angle"))

    # Adjust elements strategy if needed to enforce non-negative values
    if check_negative:
        kwargs = make_nonnegative(draw, **kwargs)

    # Generate the Parallax quantity with angle dimension
    return draw(
        ust.quantities(
            unit,
            quantity_cls=cxd.Parallax,
            check_negative=check_negative,
            **kwargs,
        )
    )
