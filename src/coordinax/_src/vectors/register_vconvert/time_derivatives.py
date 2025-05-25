"""Transformations between representations."""

__all__: list[str] = []

import functools as ft
from typing import Any

import equinox as eqx
import jax
import jax.tree as jtu
from jaxtyping import Array, Float, Real
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity, is_any_quantity

import coordinax._src.vectors.custom_types as ct
from coordinax._src.vectors import api, d1, d2, d3
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.base_acc import AbstractAcc
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel

pos_jac_fn = jax.vmap(
    jax.jacfwd(  # ↓ aux as positional argument
        lambda to_cls, from_cls, p, in_aux, out_aux: api.vconvert(
            to_cls, from_cls, p, in_aux=in_aux, out_aux=out_aux
        ),
        argnums=2,
        has_aux=True,
    ),
    in_axes=(None, None, 0, None, None),
)

is_q_or_arr = lambda x: is_any_quantity(x) or eqx.is_array(x)  # noqa: E731


def compute_jac(
    to_pos_cls: type[AbstractPos],
    from_pos_cls: type[AbstractPos],
    p_pos: ct.ParamsDict,
    /,
    *,
    in_aux: ct.AuxDict,
    out_aux: ct.AuxDict,
) -> dict[str, dict[str, u.AbstractQuantity]]:
    """Compute the Jacobian of the transformation."""
    # Compute the Jacobian of the transformation.
    # NOTE: this is using Quantities. Using raw arrays is ~20x faster.
    jac, _ = pos_jac_fn(to_pos_cls, from_pos_cls, p_pos, in_aux, out_aux)
    # Restructure the Jacobian:
    # from: ``{to_k: Quantity({from_k: Quantity(dto/dfrom, u_from)}, u_to)}``
    # to  : ``{to_k: {from_k: Quantity(dto/dfrom, u_to/u_from)}}``.
    jac = {
        out_k: {
            k: BareQuantity(v.value, out_v.unit / v.unit)
            for k, v in out_v.value.items()
        }
        for out_k, out_v in jac.items()
    }
    return jac  # noqa: RET504


@ft.partial(jax.jit, inline=True)
def inner_dot(
    inner: dict[str, u.AbstractQuantity],
    vec: dict[str, u.AbstractQuantity],
) -> u.AbstractQuantity:
    """Dot product of two dicts.

    This is a helper function for the `dot_jac_vec` function.

    Parameters
    ----------
    inner
        The first dict.
        The structure is ``{from_k: Quantity(v, unit)}``.
    vec
        The second dict.
        The structure is ``{from_k: Quantity(v, unit)}``.

    """
    return jtu.reduce(
        jnp.add,
        jtu.map(jnp.multiply, inner, vec, is_leaf=is_q_or_arr),
        is_leaf=is_q_or_arr,
    )


@jax.jit
def dot_jac_vec(
    jac: dict[str, dict[str, u.AbstractQuantity]], vec: dict[str, u.AbstractQuantity]
) -> dict[str, u.AbstractQuantity]:
    """Dot product of a Jacobian dict and a vector dict.

    This is a helper function for the `vconvert` function.

    Parameters
    ----------
    jac
        The Jacobian of the transformation.
        The structure is ``{to_k: {from_k: Quantity(v, unit)}}``.
    vec
        The vector to transform.
        The structure is ``{from_k: Quantity(v, unit)}``.

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
    # TODO: rewrite this by separating the units and the values
    return {k: inner_dot(inner, vec) for k, inner in jac.items()}


@ft.partial(jax.jit, inline=True)
def atleast_1d_float(x: Real[Array, "..."]) -> Float[Array, "..."]:
    """Convert to 1D float array."""
    return jnp.atleast_1d(jnp.astype(x, float, copy=False))


# ============================================================================


@dispatch.multi(
    (type[AbstractVel], type[AbstractVel], ct.ParamsDict, ct.ParamsDict),
    (type[AbstractAcc], type[AbstractAcc], ct.ParamsDict, ct.ParamsDict),
)
def vconvert(
    to_dif_cls: type[AbstractVector],
    from_dif_cls: type[AbstractVector],
    p_dif: ct.ParamsDict,
    p_pos: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
) -> tuple[ct.ParamsDict, ct.OptAuxDict]:
    """AbstractVel1D -> AbstractVel1D.

    Parameters
    ----------
    to_dif_cls
        The target type of the vector differential.
    from_dif_cls
        The type of the vector differential to transform.
    p_dif
        The data of the vector differential to transform.
    p_pos
        The data of the position vector used to transform the differential.
    in_aux
        The input auxiliary data to the transform.
    out_aux
        THe output auxiliary data to the transform.

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

    >>> q = {"x": u.Quantity([1.0], "km")}
    >>> a = {"x": u.Quantity([1.0], "km/s2")}
    >>> newa = cxv.vconvert(cxv.RadialAcc, cxv.CartesianAcc1D, a, q)
    >>> print(newa)
    ({'r': Quantity(Array([1.], dtype=float32), unit='km / s2')}, {})

    """
    # Check the dimensionality
    to_dif_cls = eqx.error_if(
        to_dif_cls,
        from_dif_cls._dimensionality() != to_dif_cls._dimensionality(),
        f"Dimensionality mismatch: cannot convert from {from_dif_cls.__name__} "
        f"(dim={from_dif_cls._dimensionality()}) "
        f"to {to_dif_cls.__name__} (dim={to_dif_cls._dimensionality()}) "
        "as their dimensionalities do not match.",
    )

    # Parse the auxiliary data
    in_aux_ = in_aux or {}
    out_aux_ = out_aux or {}

    # Compute the Jacobian of the position transformation.
    # The position is assumed to be in the type required by the differential to
    # construct the Jacobian. E.g. for CartesianVel1D -> RadialVel, we need the
    # Jacobian of the CartesianPos1D -> RadialPos transform.
    n = -2 if issubclass(to_dif_cls, AbstractAcc) else -1
    to_pos_cls = to_dif_cls.time_nth_derivative_cls(n=n)
    from_pos_cls = from_dif_cls.time_nth_derivative_cls(n=n)
    p_pos = jtu.map(atleast_1d_float, p_pos)
    jac = compute_jac(to_pos_cls, from_pos_cls, p_pos, in_aux=in_aux_, out_aux=out_aux_)

    # Transform the differential
    to_p_dif = dot_jac_vec(jac, p_dif)

    # Reshape the output to the shape of the input
    shape = jnp.broadcast_shapes(*[v.shape for v in p_dif.values()])
    to_p_dif = jtu.map(lambda x: jnp.reshape(x, shape), to_p_dif)

    return to_p_dif, out_aux_


# ============================================================================


# TODO: implement for cross-representations
@dispatch.multi(
    # Velocities
    (type[d1.AbstractVel1D], d1.AbstractVel1D, AbstractPos),
    (type[d2.AbstractVel2D], d2.AbstractVel2D, AbstractPos),
    (type[d3.AbstractVel3D], d3.AbstractVel3D, AbstractPos),
    # Accelerations
    (type[d1.AbstractAcc1D], d1.AbstractAcc1D, AbstractPos),
    (type[d2.AbstractAcc2D], d2.AbstractAcc2D, AbstractPos),
    (type[d3.AbstractAcc3D], d3.AbstractAcc3D, AbstractPos),
)
def vconvert(
    to_dif_cls: type[AbstractVector],
    from_dif: AbstractVector,
    from_pos: AbstractPos,
    /,
    **kwargs: Any,
) -> AbstractVector:
    """Differential -> Differential.

    This is the base case for the transformation of vector differentials.

    Parameters
    ----------
    to_dif_cls
        The target type of the vector differential.
    from_dif
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
    <RadialVel: (r) [km / s]
        [1.]>

    >>> a = cxv.CartesianAcc1D(x=u.Quantity(1.0, "km/s2"))
    >>> print(cxv.vconvert(cxv.RadialAcc, a, p, q))
    <RadialAcc: (r) [km / s2]
        [1.]>

    Now in 2D:

    >>> q = cxv.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> p = cxv.CartesianVel2D.from_([1.0, 2.0], "km/s")
    >>> newp = cxv.vconvert(cxv.PolarVel, p, q)
    >>> print(newp)
    <PolarVel: (r[km / s], phi[rad / s])
        [2.236 0.   ]>

    >>> a = cxv.CartesianAcc2D.from_([1.0, 2.0], "km/s2")
    >>> print(cxv.vconvert(cxv.PolarAcc, a, p, q))
    <PolarAcc: (r[km / s2], phi[rad / s2])
        [2.236 0.   ]>

    And in 3D:

    >>> q = cxv.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> p = cxv.CartesianVel3D.from_([1.0, 2.0, 3.0], "km/s")
    >>> newp = cxv.vconvert(cxv.SphericalVel, p, q)
    >>> print(newp.round(2))
    <SphericalVel: (r[km / s], theta[rad / s], phi[rad / s])
        [ 3.74 -0.    0.  ]>

    >>> a = cxv.CartesianAcc3D.from_([1.0, 2.0, 3.0], "km/s2")
    >>> print(cxv.vconvert(cxv.SphericalAcc, a, p, q).round(2))
    <SphericalAcc: (r[km / s2], theta[rad / s2], phi[rad / s2])
        [ 3.74 -0.    0.  ]>

    """
    # Transform the position to the type required by the differential to
    # construct the Jacobian. E.g. if we are transforming CartesianVel1D ->
    # RadialVel, we need the Jacobian of the CartesianPos1D -> RadialPos so the
    # position must be transformed to CartesianPos1D.
    n = -2 if issubclass(to_dif_cls, AbstractAcc) else -1
    in_pos_cls = from_dif.time_nth_derivative_cls(n=n)
    from_posv = vconvert(in_pos_cls, from_pos, **kwargs)

    # The position dictionary contains both the fields and the auxiliary
    # data. We need to separate them.
    pos_d = from_posv.asdict()
    in_aux = {k: pos_d.pop(k) for k in from_posv._AUX_FIELDS}

    # The output auxiliary data must either be provided through the kwargs, or
    # if the input position is the correct type, we can recover it from there.
    # For example:
    # >>> prolatespheroidalvel.vconvert(cxv.CartesianVel3D,
    # prolatespheroidalpos) Then prolatespheroidalpos has the required auxiliary
    # data. We prefer to use the kwargs, as this was used to convert from_pos to
    # from_posv.
    to_pos_cls = to_dif_cls.time_nth_derivative_cls(n=n)
    out_aux = (
        {k: getattr(from_pos, k) for k in to_pos_cls._AUX_FIELDS}
        if to_pos_cls is type(from_pos)
        else {}
    )
    out_aux |= {k: v for k, v in kwargs.items() if k in to_pos_cls._AUX_FIELDS}

    # Perform the transformation
    # TODO: add  df(q)/dt, which is 0 for all current transforms
    to_dif_p, to_dif_aux = vconvert(
        to_dif_cls,
        type(from_dif),
        from_dif.asdict(),
        pos_d,
        in_aux=in_aux,
        out_aux=out_aux,
    )

    # Reconstruct the vector
    to_dif = to_dif_cls(
        **to_dif_p,
        **{k: v for k, v in to_dif_aux.items() if k in to_dif_cls._AUX_FIELDS},
    )
    to_dif = to_dif.reshape(from_dif.shape)  # reshape to original shape

    return to_dif  # noqa: RET504


# ===================================================================


# TODO: implement for cross-representations
@dispatch.multi(
    # N-D -> N-D
    (type[d1.AbstractAcc1D], d1.AbstractAcc1D, AbstractVel, AbstractPos),
    (type[d2.AbstractAcc2D], d2.AbstractAcc2D, AbstractVel, AbstractPos),
    (type[d3.AbstractAcc3D], d3.AbstractAcc3D, AbstractVel, AbstractPos),
)
def vconvert(
    to_acc_cls: type[AbstractAcc],
    from_acc: AbstractAcc,
    from_vel: AbstractVel,
    from_pos: AbstractPos,
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
    >>> import coordinax.vecs as cxv

    Let's start in 1D:

    >>> q = cxv.CartesianPos1D(x=u.Quantity(1.0, "km"))
    >>> p = cxv.CartesianVel1D(x=u.Quantity(1.0, "km/s"))
    >>> a = cxv.CartesianAcc1D(x=u.Quantity(1.0, "km/s2"))
    >>> print(cxv.vconvert(cxv.RadialAcc, a, p, q))
    <RadialAcc: (r) [km / s2]
        [1.]>

    Now in 2D:

    >>> q = cxv.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> p = cxv.CartesianVel2D.from_([1.0, 2.0], "km/s")
    >>> a = cxv.CartesianAcc2D.from_([1.0, 2.0], "km/s2")
    >>> print(cxv.vconvert(cxv.PolarAcc, a, p, q))
    <PolarAcc: (r[km / s2], phi[rad / s2])
        [2.236 0.   ]>

    And in 3D:

    >>> q = cxv.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> p = cxv.CartesianVel3D.from_([1.0, 2.0, 3.0], "km/s")
    >>> a = cxv.CartesianAcc3D.from_([1.0, 2.0, 3.0], "km/s2")
    >>> print(cxv.vconvert(cxv.SphericalAcc, a, p, q).round(2))
    <SphericalAcc: (r[km / s2], theta[rad / s2], phi[rad / s2])
        [ 3.74 -0.    0.  ]>

    """
    del from_vel  # unused
    return vconvert(to_acc_cls, from_acc, from_pos, **kwargs)
