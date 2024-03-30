"""Transformations between representations."""

__all__ = ["represent_as"]

from math import prod
from typing import Any

import jax
from plum import dispatch

import quaxed.array_api as xp
from unxt import Quantity

from coordinax._base_dif import AbstractVectorDifferential
from coordinax._base_vec import AbstractVector
from coordinax._d1.base import Abstract1DVectorDifferential
from coordinax._d2.base import Abstract2DVectorDifferential
from coordinax._d3.base import Abstract3DVectorDifferential
from coordinax._utils import dataclass_items


# TODO: implement for cross-representations
@dispatch.multi(  # type: ignore[misc]
    # N-D -> N-D
    (
        Abstract1DVectorDifferential,
        type[Abstract1DVectorDifferential],  # type: ignore[misc]
        AbstractVector | Quantity["length"],
    ),
    (
        Abstract2DVectorDifferential,
        type[Abstract2DVectorDifferential],  # type: ignore[misc]
        AbstractVector | Quantity["length"],
    ),
    (
        Abstract3DVectorDifferential,
        type[Abstract3DVectorDifferential],  # type: ignore[misc]
        AbstractVector | Quantity["length"],
    ),
)
def represent_as(
    current: AbstractVectorDifferential,
    target: type[AbstractVectorDifferential],
    position: AbstractVector | Quantity["length"],
    /,
    **kwargs: Any,
) -> AbstractVectorDifferential:
    """AbstractVectorDifferential -> Cartesian -> AbstractVectorDifferential.

    This is the base case for the transformation of vector differentials.

    Parameters
    ----------
    current : AbstractVectorDifferential
        The vector differential to transform.
    target : type[AbstractVectorDifferential]
        The target type of the vector differential.
    position : AbstractVector
        The position vector used to transform the differential.
    **kwargs : Any
        Additional keyword arguments.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import Quantity

    Let's start in 1D:

    >>> q = cx.Cartesian1DVector(x=Quantity(1.0, "km"))
    >>> p = cx.CartesianDifferential1D(d_x=Quantity(1.0, "km/s"))
    >>> cx.represent_as(p, cx.RadialDifferential, q)
    RadialDifferential( d_r=Quantity[...]( value=f32[], unit=Unit("km / s") ) )

    Now in 2D:

    >>> q = cx.Cartesian2DVector.constructor(Quantity([1.0, 2.0], "km"))
    >>> p = cx.CartesianDifferential2D.constructor(Quantity([1.0, 2.0], "km/s"))
    >>> cx.represent_as(p, cx.PolarDifferential, q)
    PolarDifferential(
      d_r=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_phi=Quantity[...]( value=f32[], unit=Unit("rad / s") )
    )

    And in 3D:

    >>> q = cx.Cartesian3DVector.constructor(Quantity([1.0, 2.0, 3.0], "km"))
    >>> p = cx.CartesianDifferential3D.constructor(Quantity([1.0, 2.0, 3.0], "km/s"))
    >>> cx.represent_as(p, cx.SphericalDifferential, q)
    SphericalDifferential(
      d_r=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_theta=Quantity[...]( value=f32[], unit=Unit("rad / s") ),
      d_phi=Quantity[...]( value=f32[], unit=Unit("rad / s") )
    )

    If given a position as a Quantity, it will be converted to the appropriate
    Cartesian vector:

    >>> p = cx.CartesianDifferential3D.constructor(Quantity([1.0, 2.0, 3.0], "km/s"))
    >>> cx.represent_as(p, cx.SphericalDifferential, Quantity([1.0, 2.0, 3.0], "km"))
    SphericalDifferential(
      d_r=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_theta=Quantity[...]( value=f32[], unit=Unit("rad / s") ),
      d_phi=Quantity[...]( value=f32[], unit=Unit("rad / s") )
    )

    """
    # TODO: not require the shape munging / support more shapes
    shape = current.shape
    flat_shape = prod(shape)

    # Parse the position to an AbstractVector
    if isinstance(position, AbstractVector):
        posvec = position
    else:  # Q -> Cart<X>D
        posvec = current.integral_cls._cartesian_cls.constructor(  # noqa: SLF001
            position
        )

    posvec = posvec.reshape(flat_shape)  # flattened

    # Start by transforming the position to the type required by the
    # differential to construct the Jacobian.
    current_pos = represent_as(posvec, current.integral_cls, **kwargs)

    # Takes the Jacobian through the representation transformation function.  This
    # returns a representation of the target type, where the value of each field the
    # corresponding row of the Jacobian. The value of the field is a Quantity with
    # the correct numerator unit (of the Jacobian row). The value is a Vector of the
    # original type, with fields that are the columns of that row, but with only the
    # denomicator's units.
    jac_nested_vecs = jac_rep_as(current_pos, target.integral_cls)

    # This changes the Jacobian to be a dictionary of each row, with the value
    # being that row's column as a dictionary, now with the correct units for
    # each element:  {row_i: {col_j: Quantity(value, row.unit / column.unit)}}
    jac_rows = {
        f"d_{k}": {
            kk: Quantity(vv.value, unit=v.unit / vv.unit)
            for kk, vv in dataclass_items(v.value)
        }
        for k, v in dataclass_items(jac_nested_vecs)
    }

    # Now we can use the Jacobian to transform the differential.
    flat_current = current.reshape(flat_shape)
    return target(
        **{  # Each field is the dot product of the row of the J and the diff column.
            k: xp.sum(  # Doing the dot product.
                xp.stack(
                    tuple(
                        j_c * getattr(flat_current, f"d_{kk}")
                        for kk, j_c in j_r.items()
                    )
                ),
                axis=0,
            )
            for k, j_r in jac_rows.items()
        }
    ).reshape(shape)


# TODO: situate this better to show how represent_as is used
jac_rep_as = jax.jit(
    jax.vmap(jax.jacfwd(represent_as), in_axes=(0, None)), static_argnums=(1,)
)
