"""Transformations between representations."""

__all__: list[str] = []

from dataclasses import replace
from functools import partial
from math import prod
from typing import Any, cast

import equinox as eqx
import jax
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items
from unxt.quantity import is_any_quantity

from coordinax._src.distances import AbstractDistance
from coordinax._src.vectors import d1, d2, d3
from coordinax._src.vectors.base import AttrFilter
from coordinax._src.vectors.base_acc import AbstractAcc
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel

is_leaf = lambda x: is_any_quantity(x) or eqx.is_array(x)  # noqa: E731


@jax.jit
def _dot_jac_vec(
    jac: dict[str, dict[str, u.AbstractQuantity]], vec: dict[str, u.AbstractQuantity]
) -> dict[str, u.AbstractQuantity]:
    """Dot product of a Jacobian dict and a vector dict.

    This is a helper function for the `vconvert` function.

    Examples
    --------
    >>> import unxt as u

    >>> J = {"r": {"x": u.Quantity(1, ""), "y": u.Quantity(2, "")},
    ...      "phi": {"x": u.Quantity(3, "rad/km"), "y": u.Quantity(4, "rad/km")}}

    >>> v = {"x": u.Quantity(1, "km"), "y": u.Quantity(2, "km")}

    >>> _dot_jac_vec(J, v)
    {'phi': Quantity['angle'](Array(11, dtype=int32, ...), unit='rad'),
     'r': Quantity['length'](Array(5, dtype=int32, ...), unit='km')}

    """
    flat_jac, treedef = eqx.tree_flatten_one_level(jac)
    flat_dotted = jax.tree.map(
        lambda innner: jax.tree.reduce(
            jnp.add,
            jax.tree.map(jnp.multiply, innner, vec, is_leaf=is_leaf),
            is_leaf=is_leaf,
        ),
        flat_jac,
        is_leaf=lambda v: isinstance(v, dict),
    )
    return jax.tree.unflatten(treedef, flat_dotted)


# TODO: implement for cross-representations
@dispatch.multi(
    # N-D -> N-D
    (type[d1.AbstractVel1D], d1.AbstractVel1D, AbstractPos | u.Quantity["length"]),
    (type[d2.AbstractVel2D], d2.AbstractVel2D, AbstractPos | u.Quantity["length"]),
    (type[d3.AbstractVel3D], d3.AbstractVel3D, AbstractPos | u.Quantity["length"]),
)
def vconvert(
    to_vel_vector: type[AbstractVel],
    from_vel: AbstractVel,
    from_pos: AbstractPos | u.Quantity["length"],
    /,
    **kwargs: Any,
) -> AbstractVel:
    """AbstractVel -> Cartesian -> AbstractVel.

    This is the base case for the transformation of vector differentials.

    Parameters
    ----------
    to_vel_vector
        The target type of the vector differential.
    from_vel
        The vector differential to transform.
    from_pos
        The position vector used to transform the differential.
    **kwargs
        Additional keyword arguments.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Let's start in 1D:

    >>> q = cx.vecs.CartesianPos1D.from_(1.0, "km")
    >>> p = cx.vecs.CartesianVel1D.from_(1.0, "km/s")
    >>> cx.vconvert(cx.vecs.RadialVel, p, q)
    RadialVel( r=Quantity[...]( value=f32[], unit=Unit("km / s") ) )

    Now in 2D:

    >>> q = cx.vecs.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> p = cx.vecs.CartesianVel2D.from_([1.0, 2.0], "km/s")
    >>> cx.vconvert(cx.vecs.PolarVel, p, q)
    PolarVel(
      r=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      phi=Quantity[...]( value=f32[], unit=Unit("rad / s") )
    )

    And in 3D:

    >>> q = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> p = cx.CartesianVel3D.from_([1.0, 2.0, 3.0], "km/s")
    >>> cx.vconvert(cx.SphericalVel, p, q)
    SphericalVel(
      r=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      theta=Quantity[...]( value=f32[], unit=Unit("rad / s") ),
      phi=Quantity[...]( value=f32[], unit=Unit("rad / s") )
    )

    If given a position as a Quantity, it will be converted to the appropriate
    Cartesian vector:

    >>> p = cx.CartesianVel3D.from_([1.0, 2.0, 3.0], "km/s")
    >>> cx.vconvert(cx.SphericalVel, p, u.Quantity([1.0, 2.0, 3.0], "km"))
    SphericalVel(
      r=Quantity[...]( value=f32[], unit=Unit("km / s") ),
      theta=Quantity[...]( value=f32[], unit=Unit("rad / s") ),
      phi=Quantity[...]( value=f32[], unit=Unit("rad / s") )
    )

    """
    # TODO: not require the shape munging / support more shapes
    shape = from_vel.shape
    flat_shape = prod(shape)

    # ----------------------------
    # Prepare the position

    in_pos_cls = from_vel.time_antiderivative_cls
    out_pos_cls = to_vel_vector.time_antiderivative_cls

    # Parse the position to an AbstractPos
    from_posv_: AbstractPos
    if isinstance(from_pos, AbstractPos):
        from_posv_ = from_pos
    else:  # Q -> Cart<X>D
        cart_cls = in_pos_cls.cartesian_type
        from_posv_ = cast(AbstractPos, cart_cls.from_(from_pos))

    from_posv_ = from_posv_.reshape(flat_shape)  # flattened

    # Transform the position to the type required by the differential to
    # construct the Jacobian. E.g. if we are transforming CartesianVel1D ->
    # RadialVel, we need the Jacobian of the CartesianPos1D -> RadialPos so the
    # position must be transformed to CartesianPos1D.
    from_posv = vconvert(in_pos_cls, from_posv_, **kwargs)
    # The Jacobian requires the position to be a float
    dtype = jnp.result_type(float)  # TODO: better match e.g. int16 to float16?
    from_posv = from_posv.astype(dtype, copy=False)  # cast to float

    # TODO: not need to cast to distance
    from_posv = replace(
        from_posv,
        **{
            k: v.distance
            for k, v in field_items(AttrFilter, from_posv)
            if isinstance(v, AbstractDistance)
        },
    )

    # Takes the Jacobian through the representation transformation function.  This
    # returns a representation of the target type, where the value of each field the
    # corresponding row of the Jacobian. The value of the field is a Quantity with
    # the correct numerator unit (of the Jacobian row). The value is a Vector of the
    # original type, with fields that are the columns of that row, but with only the
    # denomicator's units.
    tmp = partial(vconvert, **kwargs)
    jac_rep_as = eqx.filter_jit(jax.vmap(jax.jacfwd(tmp, argnums=1), in_axes=(None, 0)))
    jac_nested_vecs = jac_rep_as(out_pos_cls, from_posv)

    # This changes the Jacobian to be a dictionary of each row, with the value
    # being that row's column as a dictionary, now with the correct units for
    # each element:  {row_i: {col_j: Quantity(value, row.unit / column.unit)}}
    jac_rows = {
        k: {
            kk: u.Quantity(vv.value, unit=v.unit / vv.unit)
            for kk, vv in field_items(AttrFilter, v.value)
        }
        for k, v in field_items(AttrFilter, jac_nested_vecs)
    }

    # Now we can use the Jacobian to transform the differential.
    flat_current = from_vel.reshape(flat_shape)
    newvec = to_vel_vector(
        **{  # Each field is the dot product of the row of the J and the diff column.
            k: jnp.sum(  # Doing the dot product.
                jnp.stack(
                    tuple(j_c * getattr(flat_current, kk) for kk, j_c in j_r.items())
                ),
                axis=0,
            )
            for k, j_r in jac_rows.items()
        }
    )

    # Reshape the output to the original shape
    newvec = newvec.reshape(shape)

    # TODO: add  df(q)/dt, which is 0 for all current transforms

    return newvec  # noqa: RET504


# TODO: implement for cross-representations
@dispatch.multi(
    # N-D -> N-D
    (
        type[d1.AbstractAcc1D],
        d1.AbstractAcc1D,
        AbstractVel | u.Quantity["speed"],
        AbstractPos | u.Quantity["length"],
    ),
    (
        type[d2.AbstractAcc2D],
        d2.AbstractAcc2D,
        AbstractVel | u.Quantity["speed"],
        AbstractPos | u.Quantity["length"],
    ),
    (
        type[d3.AbstractAcc3D],
        d3.AbstractAcc3D,
        AbstractVel | u.Quantity["speed"],
        AbstractPos | u.Quantity["length"],
    ),
)
def vconvert(
    target: type[AbstractAcc],
    current: AbstractAcc,
    velocity: AbstractVel | u.Quantity["speed"],
    position: AbstractPos | u.Quantity["length"],
    /,
    **kwargs: Any,
) -> AbstractAcc:
    """AbstractAcc -> Cartesian -> AbstractAcc.

    This is the base case for the transformation of accelerations.

    Parameters
    ----------
    current
        The vector acceleration to transform.
    target
        The target type of the vector acceleration.
    velocity
        The velocity vector used to transform the acceleration.
    position
        The position vector used to transform the acceleration.
    **kwargs
        Additional keyword arguments.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Let's start in 1D:

    >>> q = cx.vecs.CartesianPos1D(x=u.Quantity(1.0, "km"))
    >>> p = cx.vecs.CartesianVel1D(x=u.Quantity(1.0, "km/s"))
    >>> a = cx.vecs.CartesianAcc1D(x=u.Quantity(1.0, "km/s2"))
    >>> cx.vconvert(cx.vecs.RadialAcc, a, p, q)
    RadialAcc( r=Quantity[...](value=f32[], unit=Unit("km / s2")) )

    Now in 2D:

    >>> q = cx.vecs.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> p = cx.vecs.CartesianVel2D.from_([1.0, 2.0], "km/s")
    >>> a = cx.vecs.CartesianAcc2D.from_([1.0, 2.0], "km/s2")
    >>> cx.vconvert(cx.vecs.PolarAcc, a, p, q)
    PolarAcc(
      r=Quantity[...](value=f32[], unit=Unit("km / s2")),
      phi=Quantity[...]( value=f32[], unit=Unit("rad / s2") )
    )

    And in 3D:

    >>> q = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> p = cx.CartesianVel3D.from_([1.0, 2.0, 3.0], "km/s")
    >>> a = cx.vecs.CartesianAcc3D.from_([1.0, 2.0, 3.0], "km/s2")
    >>> cx.vconvert(cx.vecs.SphericalAcc, a, p, q)
    SphericalAcc(
      r=Quantity[...](value=f32[], unit=Unit("km / s2")),
      theta=Quantity[...]( value=f32[], unit=Unit("rad / s2") ),
      phi=Quantity[...]( value=f32[], unit=Unit("rad / s2") )
    )

    If given a position as a Quantity, it will be converted to the appropriate
    Cartesian vector:

    >>> cx.vconvert(cx.vecs.SphericalAcc, a,
    ...             u.Quantity([1.0, 2.0, 3.0], "km/s"),
    ...             u.Quantity([1.0, 2.0, 3.0], "km"))
    SphericalAcc(
      r=Quantity[...](value=f32[], unit=Unit("km / s2")),
      theta=Quantity[...]( value=f32[], unit=Unit("rad / s2") ),
      phi=Quantity[...]( value=f32[], unit=Unit("rad / s2") )
    )

    """
    # TODO: not require the shape munging / support more shapes
    shape = current.shape
    flat_shape = prod(shape)

    # Parse the velocity to an AbstractVel # Q -> Cart<X>D
    vel_cls = cast(AbstractVel, current.time_antiderivative_cls)
    velvec = cast(
        AbstractVel,
        velocity
        if isinstance(velocity, AbstractVel)
        else vel_cls.cartesian_type.from_(velocity),
    )

    # Parse the position to an AbstractPos (Q -> Cart<X>D)
    pos_cls = cast(AbstractPos, vel_cls.time_antiderivative_cls)
    posvec = cast(
        AbstractPos,
        position
        if isinstance(position, AbstractPos)
        else pos_cls.cartesian_type.from_(position),
    )

    posvec = posvec.reshape(flat_shape)  # flattened
    velvec = velvec.reshape(flat_shape)  # flattened

    # Start by transforming the position to the type required by the
    # differential to construct the Jacobian.
    from_posv = vconvert(current.time_nth_derivative_cls(-2), posvec, **kwargs)
    # TODO: not need to cast to distance
    from_posv = replace(
        from_posv,
        **{
            k: v.distance
            for k, v in field_items(from_posv)
            if isinstance(v, AbstractDistance)
        },
    )
    # The Jacobian requires the position to be a float
    dtype = jnp.result_type(float)  # TODO: better match e.g. int16 to float16?
    from_posv = from_posv.astype(dtype, copy=False)  # cast to float

    # Takes the Jacobian through the representation transformation function.  This
    # returns a representation of the target type, where the value of each field the
    # corresponding row of the Jacobian. The value of the field is a Quantity with
    # the correct numerator unit (of the Jacobian row). The value is a Vector of the
    # original type, with fields that are the columns of that row, but with only the
    # denomicator's units.
    jac_nested_vecs = jac_rep_as(target.time_nth_derivative_cls(-2), from_posv)

    # This changes the Jacobian to be a dictionary of each row, with the value
    # being that row's column as a dictionary, now with the correct units for
    # each element:  {row_i: {col_j: Quantity(value, row.unit / column.unit)}}
    jac_rows = {
        k: {
            kk: u.Quantity(vv.value, unit=v.unit / vv.unit)
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
                    tuple(j_c * getattr(flat_current, kk) for kk, j_c in j_r.items())
                ),
                axis=0,
            )
            for k, j_r in jac_rows.items()
        }
    ).reshape(shape)

    # TODO: add  df(q)/dt, which is 0 for all current transforms
    #       This is necessary for the df/dt * vel term in the chain rule.

    return newvec  # noqa: RET504


# TODO: situate this better to show how vconvert is used
jac_rep_as = eqx.filter_jit(
    jax.vmap(jax.jacfwd(vconvert, argnums=1), in_axes=(None, 0))
)
