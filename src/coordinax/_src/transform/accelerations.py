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

from coordinax._src.base import AbstractAcc, AbstractPos, AbstractVel
from coordinax._src.d1.base import AbstractAcc1D
from coordinax._src.d2.base import AbstractAcc2D
from coordinax._src.d3.base import AbstractAcc3D
from coordinax._src.distance.base import AbstractDistance


# TODO: implement for cross-representations
@dispatch.multi(  # type: ignore[misc]
    # N-D -> N-D
    (
        AbstractAcc1D,
        type[AbstractAcc1D],
        AbstractVel | Quantity["speed"],
        AbstractPos | Quantity["length"],
    ),
    (
        AbstractAcc2D,
        type[AbstractAcc2D],
        AbstractVel | Quantity["speed"],
        AbstractPos | Quantity["length"],
    ),
    (
        AbstractAcc3D,
        type[AbstractAcc3D],
        AbstractVel | Quantity["speed"],
        AbstractPos | Quantity["length"],
    ),
)
def represent_as(
    current: AbstractAcc,
    target: type[AbstractAcc],
    velocity: AbstractVel | Quantity["speed"],
    position: AbstractPos | Quantity["length"],
    /,
    **kwargs: Any,
) -> AbstractAcc:
    """AbstractAcc -> Cartesian -> AbstractAcc.

    This is the base case for the transformation of accelerations.

    Parameters
    ----------
    current : AbstractAcc
        The vector acceleration to transform.
    target : type[AbstractAcc]
        The target type of the vector acceleration.
    velocity : AbstractVel
        The velocity vector used to transform the acceleration.
    position : AbstractPos
        The position vector used to transform the acceleration.
    **kwargs : Any
        Additional keyword arguments.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import Quantity

    Let's start in 1D:

    >>> q = cx.CartesianPos1D(x=Quantity(1.0, "km"))
    >>> p = cx.CartesianVel1D(d_x=Quantity(1.0, "km/s"))
    >>> a = cx.CartesianAcc1D(d2_x=Quantity(1.0, "km/s2"))
    >>> cx.represent_as(a, cx.RadialAcc, p, q)
    RadialAcc( d2_r=Quantity[...](value=f32[], unit=Unit("km / s2")) )

    Now in 2D:

    >>> q = cx.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> p = cx.CartesianVel2D.from_([1.0, 2.0], "km/s")
    >>> a = cx.CartesianAcc2D.from_([1.0, 2.0], "km/s2")
    >>> cx.represent_as(a, cx.PolarAcc, p, q)
    PolarAcc(
      d2_r=Quantity[...](value=f32[], unit=Unit("km / s2")),
      d2_phi=Quantity[...]( value=f32[], unit=Unit("rad / s2") )
    )

    And in 3D:

    >>> q = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> p = cx.CartesianVel3D.from_([1.0, 2.0, 3.0], "km/s")
    >>> a = cx.CartesianAcc3D.from_([1.0, 2.0, 3.0], "km/s2")
    >>> cx.represent_as(a, cx.SphericalAcc, p, q)
    SphericalAcc(
      d2_r=Quantity[...](value=f32[], unit=Unit("km / s2")),
      d2_theta=Quantity[...]( value=f32[], unit=Unit("rad / s2") ),
      d2_phi=Quantity[...]( value=f32[], unit=Unit("rad / s2") )
    )

    If given a position as a Quantity, it will be converted to the appropriate
    Cartesian vector:

    >>> cx.represent_as(a, cx.SphericalAcc,
    ...                 Quantity([1.0, 2.0, 3.0], "km/s"),
    ...                 Quantity([1.0, 2.0, 3.0], "km"))
    SphericalAcc(
      d2_r=Quantity[...](value=f32[], unit=Unit("km / s2")),
      d2_theta=Quantity[...]( value=f32[], unit=Unit("rad / s2") ),
      d2_phi=Quantity[...]( value=f32[], unit=Unit("rad / s2") )
    )

    """
    # TODO: not require the shape munging / support more shapes
    shape = current.shape
    flat_shape = prod(shape)

    # Parse the position to an AbstractPos
    if isinstance(position, AbstractPos):
        posvec = position
    else:  # Q -> Cart<X>D
        posvec = current.integral_cls.integral_cls._cartesian_cls.from_(  # noqa: SLF001
            position
        )

    # Parse the velocity to an AbstractVel
    if isinstance(velocity, AbstractVel):
        velvec = velocity
    else:  # Q -> Cart<X>D
        velvec = current.integral_cls._cartesian_cls.from_(  # noqa: SLF001
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
            k: jnp.sum(  # Doing the dot product.
                jnp.stack(
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
jac_rep_as = eqx.filter_jit(jax.vmap(jax.jacfwd(represent_as), in_axes=(0, None)))
