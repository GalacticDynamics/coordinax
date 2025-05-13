"""Transformations between representations."""

__all__: list[str] = []

from typing import Any, cast

import equinox as eqx
import jax
import jax.tree as jtu
from jaxtyping import Array, Float, Real
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import is_any_quantity

import coordinax._src.vectors.custom_types as ct
from coordinax._src.distances import AbstractDistance
from coordinax._src.vectors import api, d1, d2, d3
from coordinax._src.vectors.base_acc import AbstractAcc
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel

# NOTE: this is using Quantities. Using raw arrays is ~20x faster.
# TODO: what about the auxiliary values?
pos_jac_fn = jax.vmap(
    jax.jacfwd(api.vconvert, argnums=2, has_aux=True), in_axes=(None, None, 0)
)

is_q_or_arr = lambda x: is_any_quantity(x) or eqx.is_array(x)  # noqa: E731


def transform_jac(
    jac: dict[str, u.AbstractQuantity],
) -> dict[str, dict[str, u.AbstractQuantity]]:
    """Transform Jacobian.

    It comes in as  ``{k: Quantity({kk: Quantity(vv, unit2)}, unit1)}``
    it needs to be rearranged to ``{k: {kk: Quantity(vv, unit1/unit2)}}``.

    """
    return {
        out_k: {
            k: u.Quantity(v.value, out_v.unit / v.unit) for k, v in out_v.value.items()
        }
        for out_k, out_v in jac.items()
    }


@jax.jit
def dot_jac_vec(
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

    >>> dot_jac_vec(J, v)
    {'phi': Quantity(Array(11, dtype=int32, ...), unit='rad'),
     'r': Quantity(Array(5, dtype=int32, ...), unit='km')}

    """
    flat_jac, treedef = eqx.tree_flatten_one_level(jac)
    flat_dotted = jtu.map(
        lambda inner: jtu.reduce(
            jnp.add,
            jtu.map(jnp.multiply, inner, vec, is_leaf=is_q_or_arr),
            is_leaf=is_q_or_arr,
        ),
        flat_jac,
        is_leaf=lambda v: isinstance(v, dict),
    )
    return jtu.unflatten(treedef, flat_dotted)


def atleast_1d_float(x: Real[Array, "..."]) -> Float[Array, "..."]:
    """Convert to 1D float array."""
    return jnp.atleast_1d(jnp.astype(x, float, copy=False))


# ============================================================================


@dispatch.multi(
    (type[AbstractVel], type[AbstractVel], ct.ParamsDict, ct.ParamsDict),
)
def vconvert(
    to_vel_cls: type[AbstractVel],
    from_vel_cls: type[AbstractVel],
    p_vel: ct.ParamsDict,
    p_pos: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
) -> tuple[ct.ParamsDict, ct.OptAuxDict]:
    """AbstractVel1D -> AbstractVel1D.

    Parameters
    ----------
    to_vel_cls
        The target type of the vector differential.
    from_vel_cls
        The type of the vector differential to transform.
    p_vel
        The data of the vector differential to transform.
    p_pos
        The data of the position vector used to transform the differential.
    in_aux
        The auxiliary data of the vector differential to transform.
    out_aux
        The auxiliary data of the position vector used to transform the
        differential.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    Let's start in 1D:

    >>> q = {"x": u.Quantity([1.0], "km")}
    >>> p = {"x": u.Quantity([1.0], "km/s")}
    >>> newp = cxv.vconvert(cxv.RadialVel, cxv.CartesianVel1D, p, q)
    >>> print(newp)
    ({'r': Quantity(Array([1.], dtype=float32), unit='km / s')}, {})

    """
    # Check the dimensionality
    to_vel_cls = eqx.error_if(
        to_vel_cls,
        from_vel_cls._dimensionality() != to_vel_cls._dimensionality(),  # noqa: SLF001
        "Dimensionality mismatch",
    )

    shape = jnp.broadcast_shapes(*[v.shape for v in p_vel.values()])

    # The position is assumed to be in the type required by the differential to
    # construct the Jacobian. E.g. for CartesianVel1D -> RadialVel, we need the
    # Jacobian of the CartesianPos1D -> RadialPos transform.
    p_pos = jtu.map(atleast_1d_float, p_pos)
    p_pos = {  # NOTE: if use unitful jacobian, this is not needed
        k: (v.distance if isinstance(v, AbstractDistance) else v)
        for k, v in p_pos.items()
    }

    # -----------------------
    # Compute the Jacobian of the position transformation.
    to_pos_cls = to_vel_cls.time_antiderivative_cls
    from_pos_cls = from_vel_cls.time_antiderivative_cls
    jac, _ = pos_jac_fn(to_pos_cls, from_pos_cls, p_pos, in_aux=in_aux)
    jac = transform_jac(jac)

    # -----------------------
    # Transform the velocity

    to_p_vel = dot_jac_vec(jac, p_vel)
    to_p_vel = jtu.map(lambda x: jnp.reshape(x, shape), to_p_vel)

    return to_p_vel, (out_aux or {})


# TODO: implement for cross-representations
@dispatch.multi(
    # N-D -> N-D
    (type[d1.AbstractVel1D], d1.AbstractVel1D, AbstractPos | u.AbstractQuantity),
    (type[d2.AbstractVel2D], d2.AbstractVel2D, AbstractPos | u.AbstractQuantity),
    (type[d3.AbstractVel3D], d3.AbstractVel3D, AbstractPos | u.AbstractQuantity),
)
def vconvert(
    to_vel_cls: type[AbstractVel],
    from_vel: AbstractVel,
    from_pos: AbstractPos | u.AbstractQuantity,
    /,
    **kwargs: Any,
) -> AbstractVel:
    """AbstractVel -> Cartesian -> AbstractVel.

    This is the base case for the transformation of vector differentials.

    Parameters
    ----------
    to_vel_cls
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
    >>> import coordinax.vecs as cxv

    Let's start in 1D:

    >>> q = cxv.CartesianPos1D.from_(1.0, "km")
    >>> p = cxv.CartesianVel1D.from_(1.0, "km/s")
    >>> newp = cxv.vconvert(cxv.RadialVel, p, q)
    >>> print(newp)
    <RadialVel (r[km / s])
        [1.]>

    Now in 2D:

    >>> q = cxv.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> p = cxv.CartesianVel2D.from_([1.0, 2.0], "km/s")
    >>> newp = cxv.vconvert(cxv.PolarVel, p, q)
    >>> print(newp)
    <PolarVel (r[km / s], phi[rad / s])
        [2.236 0.   ]>

    And in 3D:

    >>> q = cxv.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> p = cxv.CartesianVel3D.from_([1.0, 2.0, 3.0], "km/s")
    >>> newp = cxv.vconvert(cxv.SphericalVel, p, q)
    >>> print(newp)
    <SphericalVel (r[km / s], theta[rad / s], phi[rad / s])
        [ 3.742e+00 -8.941e-08  0.000e+00]>

    If given a position as a Quantity, it will be converted to the appropriate
    Cartesian vector:

    >>> p = cxv.CartesianVel3D.from_([1.0, 2.0, 3.0], "km/s")
    >>> newp = cxv.vconvert(cxv.SphericalVel, p, u.Quantity([1.0, 2.0, 3.0], "km"))
    >>> print(newp)
    <SphericalVel (r[km / s], theta[rad / s], phi[rad / s])
        [ 3.742e+00 -8.941e-08  0.000e+00]>

    """
    shape = from_vel.shape

    # ----------------------------
    # Prepare the position

    in_pos_cls = from_vel.time_antiderivative_cls

    # Parse the position to an AbstractPos
    from_posv_ = cast(
        AbstractPos,
        from_pos
        if isinstance(from_pos, AbstractPos)
        else in_pos_cls.cartesian_type.from_(from_pos),
    )

    # Transform the position to the type required by the differential to
    # construct the Jacobian. E.g. if we are transforming CartesianVel1D ->
    # RadialVel, we need the Jacobian of the CartesianPos1D -> RadialPos so the
    # position must be transformed to CartesianPos1D.
    from_posv = vconvert(in_pos_cls, from_posv_, **kwargs)

    # ----------------------------
    # Perform the transformation

    # TODO: add  df(q)/dt, which is 0 for all current transforms
    to_vel_params, to_vel_aux = vconvert(
        to_vel_cls,
        type(from_vel),
        from_vel.asdict(),
        from_posv.asdict(),
    )

    # ----------------------------
    # Reconstruct the vector

    to_vel = to_vel_cls(**to_vel_params, **to_vel_aux)
    to_vel = to_vel.reshape(shape)  # reshape to original shape

    return to_vel  # noqa: RET504


# ===================================================================


@dispatch
def vconvert(
    to_acc_cls: type[AbstractAcc],
    from_acc_cls: type[AbstractAcc],
    p_acc: ct.ParamsDict,
    p_vel: ct.ParamsDict,
    p_pos: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
) -> tuple[ct.ParamsDict, ct.OptAuxDict]:
    """AbstractAcc1D -> AbstractAcc1D.

    Parameters
    ----------
    to_acc_cls
        The target type of the vector differential.
    from_acc_cls
        The type of the vector differential to transform.
    p_acc
        The data of the vector differential to transform.
    p_vel, p_pos
        The data of the velocity/position vector used to transform the
        differential.
    in_aux
        The auxiliary data of the vector differential to transform.
    out_aux
        The auxiliary data of the position vector used to transform the
        differential.
    units
        The unit system to use for the transformation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    Let's start in 1D:

    >>> q = {"x": u.Quantity([1.0], "km")}
    >>> p = {"x": u.Quantity([1.0], "km/s")}
    >>> a = {"x": u.Quantity([1.0], "km/s2")}
    >>> newa = cxv.vconvert(cxv.RadialAcc, cxv.CartesianAcc1D, a, p, q)
    >>> print(newa)
    ({'r': Quantity(Array([1.], dtype=float32), unit='km / s2')}, {})

    """
    # Check the dimensionality
    to_acc_cls = eqx.error_if(
        to_acc_cls,
        from_acc_cls._dimensionality() != to_acc_cls._dimensionality(),  # noqa: SLF001
        "Dimensionality mismatch",
    )

    shape = jnp.broadcast_shapes(*[v.shape for v in p_acc.values()])

    # The position is assumed to be in the type required by the differential to
    # construct the Jacobian. E.g. for CartesianVel1D -> RadialVel, we need the
    # Jacobian of the CartesianPos1D -> RadialPos transform.
    p_pos = jtu.map(atleast_1d_float, p_pos)
    p_pos = {  # NOTE: if use unitful jacobian, this is not needed
        k: (v.distance if isinstance(v, AbstractDistance) else v)
        for k, v in p_pos.items()
    }

    # -----------------------
    # Compute the Jacobian of the position transformation.
    to_pos_cls = to_acc_cls.time_nth_derivative_cls(-2)
    from_pos_cls = from_acc_cls.time_nth_derivative_cls(-2)
    jac, _ = pos_jac_fn(to_pos_cls, from_pos_cls, p_pos, in_aux=in_aux)
    jac = transform_jac(jac)

    # -----------------------
    # Transform the acceleration

    to_p_acc = dot_jac_vec(jac, p_acc)
    to_p_acc = jtu.map(lambda x: jnp.reshape(x, shape), to_p_acc)

    return to_p_acc, (out_aux or {})


# TODO: implement for cross-representations
@dispatch.multi(
    # N-D -> N-D
    (
        type[d1.AbstractAcc1D],
        d1.AbstractAcc1D,
        AbstractVel | u.AbstractQuantity,
        AbstractPos | u.AbstractQuantity,
    ),
    (
        type[d2.AbstractAcc2D],
        d2.AbstractAcc2D,
        AbstractVel | u.AbstractQuantity,
        AbstractPos | u.AbstractQuantity,
    ),
    (
        type[d3.AbstractAcc3D],
        d3.AbstractAcc3D,
        AbstractVel | u.AbstractQuantity,
        AbstractPos | u.AbstractQuantity,
    ),
)
def vconvert(
    to_acc_cls: type[AbstractAcc],
    from_acc: AbstractAcc,
    from_vel: AbstractVel | u.Quantity["speed"],
    from_pos: AbstractPos | u.Quantity["length"],
    /,
    **kwargs: Any,
) -> AbstractAcc:
    r"""AbstractAcc -> Cartesian -> AbstractAcc.

    This is the base case for the transformation of accelerations.

    Let $\mathbf{x}$ be a position vector in one representation $\mathbf{y}$ a
    position vector in another representation related by:

    $$ \mathbf{y} = f(\mathbf{x}) $$

    where $f$ is a differentiable function mapping between the coordinate
    systems.

    The Jacobian matrix $J$ of the transformation is:

    $$ J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$$

    The coordinate transformation of the acceleration is given by the chain
    rule:

    $$ \ddot{\mathbf{y}} = \dot{J} \dot{\mathbf{x}} + J \ddot{\mathbf{x}}$$

    where $\dot{J}$ is the time derivative of the Jacobian matrix. This function
    assumes that the representation conversion is time-invariant, so $\dot{J} =
    0$. Thus, the transformation simplifies to:

    $$ \ddot{\mathbf{y}} = J \ddot{\mathbf{x}}$$

    This function implements this transformation.

    Parameters
    ----------
    to_acc_cls
        The target type of the vector acceleration.
    from_acc
        The vector acceleration to transform.
    from_vel
        The velocity vector used to transform the acceleration.
    from_pos
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
    RadialAcc(r=Quantity(f32[], unit='km / s2'))

    Now in 2D:

    >>> q = cx.vecs.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> p = cx.vecs.CartesianVel2D.from_([1.0, 2.0], "km/s")
    >>> a = cx.vecs.CartesianAcc2D.from_([1.0, 2.0], "km/s2")
    >>> cx.vconvert(cx.vecs.PolarAcc, a, p, q)
    PolarAcc(
      r=Quantity(f32[], unit='km / s2'), phi=Quantity(f32[], unit='rad / s2')
    )

    And in 3D:

    >>> q = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> p = cx.CartesianVel3D.from_([1.0, 2.0, 3.0], "km/s")
    >>> a = cx.vecs.CartesianAcc3D.from_([1.0, 2.0, 3.0], "km/s2")
    >>> cx.vconvert(cx.vecs.SphericalAcc, a, p, q)
    SphericalAcc(
      r=Quantity(f32[], unit='km / s2'),
      theta=Quantity(f32[], unit='rad / s2'),
      phi=Quantity(f32[], unit='rad / s2')
    )

    If given a position as a Quantity, it will be converted to the appropriate
    Cartesian vector:

    >>> cx.vconvert(cx.vecs.SphericalAcc, a,
    ...             u.Quantity([1.0, 2.0, 3.0], "km/s"),
    ...             u.Quantity([1.0, 2.0, 3.0], "km"))
    SphericalAcc(
      r=Quantity(f32[], unit='km / s2'),
      theta=Quantity(f32[], unit='rad / s2'),
      phi=Quantity(f32[], unit='rad / s2')
    )

    """
    shape = from_acc.shape

    # ----------------------------
    # Prepare the position

    # Parse the position to an AbstractPos
    in_pos_cls = from_acc.time_nth_derivative_cls(-2)
    from_posv_ = cast(
        AbstractPos,
        from_pos
        if isinstance(from_pos, AbstractPos)
        else in_pos_cls.cartesian_type.from_(from_pos),
    )

    # Transform the position to the type required by the differential to
    # construct the Jacobian. E.g. if we are transforming CartesianVel1D ->
    # RadialVel, we need the Jacobian of the CartesianPos1D -> RadialPos so the
    # position must be transformed to CartesianPos1D.
    from_posv = vconvert(in_pos_cls, from_posv_, **kwargs)

    # ----------------------------
    # Prepare the velocity

    in_vel_cls = from_acc.time_antiderivative_cls
    from_velv_ = cast(
        AbstractVel,
        from_vel
        if isinstance(from_vel, AbstractVel)
        else in_vel_cls.cartesian_type.from_(from_vel),
    )
    from_velv = vconvert(in_vel_cls, from_velv_, from_posv, **kwargs)

    # ----------------------------
    # Perform the transformation

    # TODO: add  df(q)/dt, which is 0 for all current transforms
    to_acc_params, to_acc_aux = vconvert(
        to_acc_cls,
        type(from_acc),
        from_acc.asdict(),
        from_velv.asdict(),
        from_posv.asdict(),
    )

    # ----------------------------
    # Reconstruct the vector

    to_acc = to_acc_cls(**to_acc_params, **to_acc_aux)
    to_acc = to_acc.reshape(shape)  # reshape to original shape

    return to_acc  # noqa: RET504
