"""Hypothesis strategies for Distance quantities."""

__all__ = ("distance_moduli",)

import warnings

from typing import Any, cast

from hypothesis import strategies as st

import unxt_hypothesis as ust

import coordinax.distances as cxd


@st.composite
def distance_moduli(draw: st.DrawFn, /, **kwargs: Any) -> cxd.DistanceModulus:
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
    >>> import coordinax_hypothesis.core as cxst
    >>> import coordinax.distances as cxd

    >>> @given(dm=cxst.distance_moduli())
    ... def test_distance_modulus(dm):
    ...     assert isinstance(dm, cxd.DistanceModulus)
    ...     assert dm.unit == "mag"

    Generate distance modulus arrays:

    >>> @given(dm=cxst.distance_moduli(shape=10))
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
    out = draw(ust.quantities(unit="mag", quantity_cls=cxd.DistanceModulus, **kwargs))
    return cast("cxd.DistanceModulus", out)


# Register type strategy for Hypothesis's st.from_type()
# Note: Pass the callable, not an invoked strategy
st.register_type_strategy(cxd.DistanceModulus, lambda _: distance_moduli())
