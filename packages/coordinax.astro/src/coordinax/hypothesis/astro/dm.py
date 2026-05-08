"""Hypothesis strategies for Distance quantities."""

__all__ = ("distance_moduli",)

import warnings

from typing import Any, cast

from hypothesis import strategies as st

import unxt_hypothesis as ust

import coordinax.astro as cxastro


@st.composite
def distance_moduli(draw: st.DrawFn, /, **kwargs: Any) -> cxastro.DistanceModulus:
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
    coordinax.astro.DistanceModulus
        A strategy that generates DistanceModulus instances.

    Warns
    -----
    UserWarning
        If 'unit' is specified in kwargs, since DistanceModulus always uses
        'mag' units and the argument will be ignored.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.hypothesis.astro as cxastrost
    >>> import coordinax.astro as cxastro

    >>> @given(dm=cxastrost.distance_moduli())
    ... def test_distance_modulus(dm):
    ...     assert isinstance(dm, cxastro.DistanceModulus)
    ...     assert dm.unit == "mag"

    Generate distance modulus arrays:

    >>> @given(dm=cxastrost.distance_moduli(shape=10))
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
    out = draw(
        ust.quantities(unit="mag", quantity_cls=cxastro.DistanceModulus, **kwargs)
    )
    return cast("cxastro.DistanceModulus", out)


# Register type strategy for Hypothesis's st.from_type()
# Note: Pass the callable, not an invoked strategy
st.register_type_strategy(cxastro.DistanceModulus, lambda _: distance_moduli())
