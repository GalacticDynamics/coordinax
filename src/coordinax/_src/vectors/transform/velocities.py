"""Transformations between representations."""

__all__: list[str] = []

from dataclasses import replace
from functools import partial
from math import prod
from typing import Any

import equinox as eqx
import jax
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items

from coordinax._src.distances import AbstractDistance
from coordinax._src.vectors.base import AbstractPos, AbstractVel
from coordinax._src.vectors.base.flags import AttrFilter
from coordinax._src.vectors.d1 import AbstractVel1D
from coordinax._src.vectors.d2 import AbstractVel2D
from coordinax._src.vectors.d3 import AbstractVel3D


# TODO: implement for cross-representations
@dispatch.multi(  # type: ignore[misc]
    # N-D -> N-D
    (type[AbstractVel1D], AbstractVel1D, AbstractPos | u.Quantity["length"]),
    (type[AbstractVel2D], AbstractVel2D, AbstractPos | u.Quantity["length"]),
    (type[AbstractVel3D], AbstractVel3D, AbstractPos | u.Quantity["length"]),
)
def vconvert(
    target: type[AbstractVel],
    current: AbstractVel,
    position: AbstractPos | u.Quantity["length"],
    /,
    **kwargs: Any,
) -> AbstractVel:
    """AbstractVel -> Cartesian -> AbstractVel.

    This is the base case for the transformation of vector differentials.

    Parameters
    ----------
    target : type[AbstractVel]
        The target type of the vector differential.
    current : AbstractVel
        The vector differential to transform.
    position : AbstractPos
        The position vector used to transform the differential.
    **kwargs : Any
        Additional keyword arguments.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Let's start in 1D:

    >>> q = cx.vecs.CartesianPos1D.from_(1.0, "km")
    >>> p = cx.vecs.CartesianVel1D.from_(1.0, "km/s")
    >>> cx.vconvert(cx.vecs.RadialVel, p, q)
    RadialVel( d_r=Quantity[...]( value=f32[], unit=Unit("km / s") ) )

    Now in 2D:

    >>> q = cx.vecs.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> p = cx.vecs.CartesianVel2D.from_([1.0, 2.0], "km/s")
    >>> cx.vconvert(cx.vecs.PolarVel, p, q)
    PolarVel(
      d_r=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_phi=Quantity[...]( value=f32[], unit=Unit("rad / s") )
    )

    And in 3D:

    >>> q = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> p = cx.CartesianVel3D.from_([1.0, 2.0, 3.0], "km/s")
    >>> cx.vconvert(cx.SphericalVel, p, q)
    SphericalVel(
      d_r=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      d_theta=Quantity[...]( value=f32[], unit=Unit("rad / s") ),
      d_phi=Quantity[...]( value=f32[], unit=Unit("rad / s") )
    )

    If given a position as a Quantity, it will be converted to the appropriate
    Cartesian vector:

    >>> p = cx.CartesianVel3D.from_([1.0, 2.0, 3.0], "km/s")
    >>> cx.vconvert(cx.SphericalVel, p, u.Quantity([1.0, 2.0, 3.0], "km"))
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
    current_pos = vconvert(current.integral_cls, posvec, **kwargs)
    # TODO: not need to cast to distance
    current_pos = replace(
        current_pos,
        **{
            k: v.distance
            for k, v in field_items(AttrFilter, current_pos)
            if isinstance(v, AbstractDistance)
        },
    )
    # The Jacobian requires the position to be a float
    dtype = jnp.result_type(float)  # TODO: better match e.g. int16 to float16?
    current_pos = current_pos.astype(dtype, copy=False)  # cast to float

    # Takes the Jacobian through the representation transformation function.  This
    # returns a representation of the target type, where the value of each field the
    # corresponding row of the Jacobian. The value of the field is a Quantity with
    # the correct numerator unit (of the Jacobian row). The value is a Vector of the
    # original type, with fields that are the columns of that row, but with only the
    # denomicator's units.
    tmp = partial(vconvert, **kwargs)
    jac_rep_as = eqx.filter_jit(jax.vmap(jax.jacfwd(tmp, argnums=1), in_axes=(None, 0)))
    jac_nested_vecs = jac_rep_as(target.integral_cls, current_pos)

    # This changes the Jacobian to be a dictionary of each row, with the value
    # being that row's column as a dictionary, now with the correct units for
    # each element:  {row_i: {col_j: Quantity(value, row.unit / column.unit)}}
    jac_rows = {
        f"d_{k}": {
            kk: u.Quantity(vv.value, unit=v.unit / vv.unit)
            for kk, vv in field_items(AttrFilter, v.value)
        }
        for k, v in field_items(AttrFilter, jac_nested_vecs)
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
