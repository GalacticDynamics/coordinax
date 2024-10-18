"""Transformations between representations."""

__all__: list[str] = []

from dataclasses import replace
from math import prod
from typing import Any

import equinox as eqx
import jax
from plum import dispatch

import quaxed.numpy as jnp
from dataclassish import field_items
from unxt import Quantity

from coordinax._src.base import AbstractPos, AbstractVel
from coordinax._src.d1.base import AbstractVel1D
from coordinax._src.d2.base import AbstractVel2D
from coordinax._src.d3.base import AbstractVel3D
from coordinax._src.distance.base import AbstractDistance


# TODO: implement for cross-representations
@dispatch.multi(  # type: ignore[misc]
    # N-D -> N-D
    (AbstractVel1D, type[AbstractVel1D], AbstractPos | Quantity["length"]),
    (AbstractVel2D, type[AbstractVel2D], AbstractPos | Quantity["length"]),
    (AbstractVel3D, type[AbstractVel3D], AbstractPos | Quantity["length"]),
)
def represent_as(
    current: AbstractVel,
    target: type[AbstractVel],
    position: AbstractPos | Quantity["length"],
    /,
    **kwargs: Any,
) -> AbstractVel:
    """AbstractVel -> Cartesian -> AbstractVel.

    This is the base case for the transformation of vector differentials.

    Parameters
    ----------
    current : AbstractVel
        The vector differential to transform.
    target : type[AbstractVel]
        The target type of the vector differential.
    position : AbstractPos
        The position vector used to transform the differential.
    **kwargs : Any
        Additional keyword arguments.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import Quantity

    Let's start in 1D:

    >>> q = cx.CartesianPos1D(x=Quantity(1.0, "km"))
    >>> p = cx.CartesianVel1D(d_x=Quantity(1.0, "km/s"))
    >>> cx.represent_as(p, cx.RadialVel, q)
    RadialVel( d_r=Quantity[...]( value=f32[], unit=Unit("km / s") ) )

    Now in 2D:

    >>> q = cx.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> p = cx.CartesianVel2D.from_([1.0, 2.0], "km/s")
    >>> cx.represent_as(p, cx.PolarVel, q)
    PolarVel(
      d_r=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_phi=Quantity[...]( value=f32[], unit=Unit("rad / s") )
    )

    And in 3D:

    >>> q = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> p = cx.CartesianVel3D.from_([1.0, 2.0, 3.0], "km/s")
    >>> cx.represent_as(p, cx.SphericalVel, q)
    SphericalVel(
      d_r=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_theta=Quantity[...]( value=f32[], unit=Unit("rad / s") ),
      d_phi=Quantity[...]( value=f32[], unit=Unit("rad / s") )
    )

    If given a position as a Quantity, it will be converted to the appropriate
    Cartesian vector:

    >>> p = cx.CartesianVel3D.from_([1.0, 2.0, 3.0], "km/s")
    >>> cx.represent_as(p, cx.SphericalVel, Quantity([1.0, 2.0, 3.0], "km"))
    SphericalVel(
      d_r=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_theta=Quantity[...]( value=f32[], unit=Unit("rad / s") ),
      d_phi=Quantity[...]( value=f32[], unit=Unit("rad / s") )
    )

    """
    # TODO: not require the shape munging / support more shapes
    shape = current.shape
    flat_shape = prod(shape)

    # Parse the position to an AbstractPos
    if isinstance(position, AbstractPos):
        posvec = position
    else:  # Q -> Cart<X>D
        posvec = current.integral_cls._cartesian_cls.from_(  # noqa: SLF001
            position
        )

    posvec = posvec.reshape(flat_shape)  # flattened

    # Start by transforming the position to the type required by the
    # differential to construct the Jacobian.
    current_pos = represent_as(posvec, current.integral_cls, **kwargs)
    # TODO: not need to cast to distance
    current_pos = replace(
        current_pos,
        **{
            k: v.distance
            for k, v in field_items(current_pos)
            if isinstance(v, AbstractDistance)
        },
    )

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
            for kk, vv in field_items(v.value)
        }
        for k, v in field_items(jac_nested_vecs)
    }

    # Now we can use the Jacobian to transform the differential.
    flat_current = current.reshape(flat_shape)
    newvec = target(
        **{  # Each field is the dot product of the row of the J and the diff column.
            k: jnp.sum(  # Doing the dot product.
                jnp.stack(
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

    # TODO: add  df(q)/dt, which is 0 for all current transforms

    return newvec  # noqa: RET504


# TODO: situate this better to show how represent_as is used
jac_rep_as = eqx.filter_jit(jax.vmap(jax.jacfwd(represent_as), in_axes=(0, None)))
