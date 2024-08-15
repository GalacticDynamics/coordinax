"""Transformations between representations."""

__all__ = ["represent_as"]

from dataclasses import replace
from math import prod
from typing import Any

import jax
from plum import dispatch

import quaxed.array_api as xp
from dataclassish import field_items
from unxt import AbstractDistance, Quantity

from coordinax._coordinax.base_acc import AbstractAcceleration
from coordinax._coordinax.base_pos import AbstractPosition
from coordinax._coordinax.base_vel import AbstractVelocity
from coordinax._coordinax.d1.base import AbstractAcceleration1D
from coordinax._coordinax.d2.base import AbstractAcceleration2D
from coordinax._coordinax.d3.base import AbstractAcceleration3D


# TODO: implement for cross-representations
@dispatch.multi(  # type: ignore[misc]
    # N-D -> N-D
    (
        AbstractAcceleration1D,
        type[AbstractAcceleration1D],
        AbstractVelocity | Quantity["speed"],
        AbstractPosition | Quantity["length"],
    ),
    (
        AbstractAcceleration2D,
        type[AbstractAcceleration2D],
        AbstractVelocity | Quantity["speed"],
        AbstractPosition | Quantity["length"],
    ),
    (
        AbstractAcceleration3D,
        type[AbstractAcceleration3D],
        AbstractVelocity | Quantity["speed"],
        AbstractPosition | Quantity["length"],
    ),
)
def represent_as(
    current: AbstractAcceleration,
    target: type[AbstractAcceleration],
    velocity: AbstractVelocity | Quantity["speed"],
    position: AbstractPosition | Quantity["length"],
    /,
    **kwargs: Any,
) -> AbstractAcceleration:
    """AbstractAcceleration -> Cartesian -> AbstractAcceleration.

    This is the base case for the transformation of accelerations.

    Parameters
    ----------
    current : AbstractAcceleration
        The vector acceleration to transform.
    target : type[AbstractAcceleration]
        The target type of the vector acceleration.
    velocity : AbstractVelocity
        The velocity vector used to transform the acceleration.
    position : AbstractPosition
        The position vector used to transform the acceleration.
    **kwargs : Any
        Additional keyword arguments.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import Quantity

    Let's start in 1D:

    >>> q = cx.CartesianPosition1D(x=Quantity(1.0, "km"))
    >>> p = cx.CartesianVelocity1D(d_x=Quantity(1.0, "km/s"))
    >>> a = cx.CartesianAcceleration1D(d2_x=Quantity(1.0, "km/s2"))
    >>> cx.represent_as(a, cx.RadialAcceleration, p, q)
    RadialAcceleration( d2_r=Quantity[...](value=f32[], unit=Unit("km / s2")) )

    Now in 2D:

    >>> q = cx.CartesianPosition2D.constructor([1.0, 2.0], "km")
    >>> p = cx.CartesianVelocity2D.constructor([1.0, 2.0], "km/s")
    >>> a = cx.CartesianAcceleration2D.constructor([1.0, 2.0], "km/s2")
    >>> cx.represent_as(a, cx.PolarAcceleration, p, q)
    PolarAcceleration(
      d2_r=Quantity[...](value=f32[], unit=Unit("km / s2")),
      d2_phi=Quantity[...]( value=f32[], unit=Unit("rad / s2") )
    )

    And in 3D:

    >>> q = cx.CartesianPosition3D.constructor([1.0, 2.0, 3.0], "km")
    >>> p = cx.CartesianVelocity3D.constructor([1.0, 2.0, 3.0], "km/s")
    >>> a = cx.CartesianAcceleration3D.constructor([1.0, 2.0, 3.0], "km/s2")
    >>> cx.represent_as(a, cx.SphericalAcceleration, p, q)
    SphericalAcceleration(
      d2_r=Quantity[...](value=f32[], unit=Unit("km / s2")),
      d2_theta=Quantity[...]( value=f32[], unit=Unit("rad / s2") ),
      d2_phi=Quantity[...]( value=f32[], unit=Unit("rad / s2") )
    )

    If given a position as a Quantity, it will be converted to the appropriate
    Cartesian vector:

    >>> cx.represent_as(a, cx.SphericalAcceleration,
    ...                 Quantity([1.0, 2.0, 3.0], "km/s"),
    ...                 Quantity([1.0, 2.0, 3.0], "km"))
    SphericalAcceleration(
      d2_r=Quantity[...](value=f32[], unit=Unit("km / s2")),
      d2_theta=Quantity[...]( value=f32[], unit=Unit("rad / s2") ),
      d2_phi=Quantity[...]( value=f32[], unit=Unit("rad / s2") )
    )

    """
    # TODO: not require the shape munging / support more shapes
    shape = current.shape
    flat_shape = prod(shape)

    # Parse the position to an AbstractPosition
    if isinstance(position, AbstractPosition):
        posvec = position
    else:  # Q -> Cart<X>D
        posvec = current.integral_cls.integral_cls._cartesian_cls.constructor(  # noqa: SLF001
            position
        )

    # Parse the velocity to an AbstractVelocity
    if isinstance(velocity, AbstractVelocity):
        velvec = velocity
    else:  # Q -> Cart<X>D
        velvec = current.integral_cls._cartesian_cls.constructor(  # noqa: SLF001
            velocity
        )

    posvec = posvec.reshape(flat_shape)  # flattened
    velvec = velvec.reshape(flat_shape)  # flattened

    # Start by transforming the position to the type required by the
    # differential to construct the Jacobian.
    current_pos = represent_as(posvec, current.integral_cls.integral_cls, **kwargs)
    # TODO: not need to cast to distance
    current_pos = replace(
        current_pos,
        **{
            k: v.distance
            for k, v in field_items(current_pos)
            if isinstance(v, AbstractDistance)
        },
    )

    # Start by transforming the velocity to the type required by the
    # differential to construct the Jacobian.
    current_vel = represent_as(velvec, current.integral_cls, current_pos, **kwargs)  # noqa: F841  # pylint: disable=unused-variable

    # Takes the Jacobian through the representation transformation function.  This
    # returns a representation of the target type, where the value of each field the
    # corresponding row of the Jacobian. The value of the field is a Quantity with
    # the correct numerator unit (of the Jacobian row). The value is a Vector of the
    # original type, with fields that are the columns of that row, but with only the
    # denomicator's units.
    jac_nested_vecs = jac_rep_as(current_pos, target.integral_cls.integral_cls)

    # This changes the Jacobian to be a dictionary of each row, with the value
    # being that row's column as a dictionary, now with the correct units for
    # each element:  {row_i: {col_j: Quantity(value, row.unit / column.unit)}}
    jac_rows = {
        f"d2_{k}": {
            kk: Quantity(vv.value, unit=v.unit / vv.unit)
            for kk, vv in field_items(v.value)
        }
        for k, v in field_items(jac_nested_vecs)
    }

    # Now we can use the Jacobian to transform the differential.
    flat_current = current.reshape(flat_shape)
    newvec = target(
        **{  # Each field is the dot product of the row of the J and the diff column.
            k: xp.sum(  # Doing the dot product.
                xp.stack(
                    tuple(
                        j_c * getattr(flat_current, f"d2_{kk}")
                        for kk, j_c in j_r.items()
                    )
                ),
                axis=0,
            )
            for k, j_r in jac_rows.items()
        }
    ).reshape(shape)

    # TODO: add  df(q)/dt, which is 0 for all current transforms
    #       This is necessary for the df/dt * vel term in the chain rule.

    return newvec  # noqa: RET504


# TODO: situate this better to show how represent_as is used
jac_rep_as = jax.jit(
    jax.vmap(jax.jacfwd(represent_as), in_axes=(0, None)), static_argnums=(1,)
)
