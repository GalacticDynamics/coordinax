"""Galilean coordinate transformations."""
from jax.typing import ArrayLike

__all__ = ("Rotate",)


from dataclasses import replace

from collections.abc import Callable
from jaxtyping import Array, Shaped
from typing import Any, Final, TypeAlias, final, get_type_hints

import equinox as eqx
import jax
import plum
from jax.scipy.spatial.transform import Rotation
from quax import quaxify

import quaxed.numpy as jnp
import unxt as u

from .base import AbstractOperator, Neg, eval_op
from coordinax._src import api, charts as cxc, roles as cxr
from coordinax._src.custom_types import CsDict
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.identity import Identity
from coordinax._src.transformations import frames as cxf
from coordinax._src.transformations.register_physicaldiff_map import (
    _pack_uniform_unit,
    _unpack_with_unit,
)

vec_matmul = quaxify(jax.numpy.vectorize(jax.numpy.matmul, signature="(3,3),(3)->(3)"))

RMatrix: TypeAlias = Shaped[Array, "3 3"]


@final
class Rotate(AbstractOperator):
    r"""Operator for Galilean rotations.

    The coordinate transform is given by:

    $$
    $$
        (t,\mathbf{x}) \mapsto (t, R \mathbf{x})

    where $R$ is the rotation matrix.  Note this is intrinsically time
    dependent.

    Parameters
    ----------
    rotation : Array[float, (3, 3)]
        The rotation matrix.

    Raises
    ------
    ValueError
        If the rotation matrix is not orthogonal.

    Notes
    -----
    The Galilean rotation is intrinsically a time-dependent transformation.
    This is part of the inhomogeneous Galilean group, which is the group of
    transformations that leave the space-time interval invariant.

    Examples
    --------
    We start with the required imports:

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    We can then create a time-invariant rotation operator:

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0,  0], [0, 0, 1]])
    >>> op = cx.ops.Rotate(Rz)
    >>> op
    Rotate(i64[3,3](jax))

    Rotation operators can be applied to a `unxt.Quantity`, taken to represent a
    Cartesian vector:

    >>> q = u.Q([1, 0, 0], "m")
    >>> t = u.Q(1, "s")
    >>> op(t, q)
    Quantity(Array([0, 1, 0], dtype=int64), unit='m')

    This also works for a batch of vectors:

    >>> q = u.Q([[1, 0, 0], [0, 1, 0]], "m")
    >>> op(t, q)
    Quantity(Array([[ 0,  1,  0],
                    [-1,  0,  0]], dtype=int64), unit='m')

    Rotation operators can be applied to `coordinax.Vector`:

    >>> q = cx.Vector.from_(q)  # from the previous example
    >>> op(t, q)
    Quantity(Array([ 0, -1], dtype=int32), unit='m')

    You can make the rotation matrix time-dependent:

    >>> from jaxtyping import Array, Real
    >>> def R_func(t) -> Real[Array, "3 3"]:
    ...     theta = (jnp.pi / 4) * t.to_value("s")
    ...     st, ct = jnp.sin(theta), jnp.cos(theta)
    ...     return jnp.array([[ct, -st, 0], [st,  ct, 0], [0, 0, 1]])

    >>> R_op = cx.ops.Rotate.from_(R_func)
    >>> R_op

    >>> t = u.Q(2, "s")
    >>> op(t, q)

    """

    R: Shaped[Array, " N N"] | Callable[[Any], RMatrix]
    """The rotation vector."""

    # -----------------------------------------------------

    @classmethod
    def from_euler(
        cls: type["Rotate"], seq: str, angles: u.Q["angle"] | u.Angle, /
    ) -> "Rotate":
        """Initialize from Euler angles.

        See `jax.scipy.spatial.transform.Rotation.from_euler`.
        `XYZ` are intrinsic rotations, `xyz` are extrinsic rotations.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> op = cx.ops.Rotate.from_euler("z", u.Q(90, "deg"))
        >>> op.R.round(2)
        Array([[ 0., -1.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  0.,  1.]], dtype=float32)

        """
        R = Rotation.from_euler(seq, u.ustrip("deg", angles), degrees=True).as_matrix()
        return cls(R)

    @classmethod
    @AbstractOperator.from_.dispatch  # type: ignore[untyped-decorator]
    def from_(cls: type["Rotate"], obj: Rotation, /) -> "Rotate":
        """Initialize from a `jax.scipy.spatial.transform.Rotation`.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from jax.scipy.spatial.transform import Rotation
        >>> import coordinax as cx

        >>> R = Rotation.from_euler("z", 90, degrees=True)
        >>> op = cx.ops.Rotate.from_(R)

        >>> jnp.allclose(op.R, R.as_matrix())
        Array(True, dtype=bool)

        """
        return cls(obj.as_matrix())

    # -----------------------------------------------------

    @classmethod
    def operate(cls, params: dict[str, Any], arg: Any, /, **__: Any) -> Any:
        """Apply the :class:`coordinax.ops.Rotate` operation.

        This is the identity operation, which does nothing to the input.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.ops as cxo

        >>> q = u.Q([1, 2, 3], "km")
        >>> R = jnp.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        >>> cxo.operate(cxo.Rotate, {"R": R}, q)

        >>> vec = cxo.Cart3D.from_([1, 2, 3], "km")
        >>> cxo.operate(cxo.Rotate, {"R": R}, vec)

        """
        return params["R"] @ arg
        # return vec_matmul(params["R"], arg)

    @property
    def inverse(self) -> "Rotate":
        """The inverse of the operator.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import coordinax as cx

        >>> Rz = jnp.asarray([[0, -1, 0], [1, 0,  0], [0, 0, 1]])
        >>> op = cx.ops.Rotate(Rz)
        >>> op.inverse
        Rotate(R=i32[3,3])

        >>> jnp.allclose(op.R, op.inverse.R.T)
        Array(True, dtype=bool)

        """
        R = self.R
        return replace(  # TODO: a transposition wrapper
            self, R=R.T if not callable(R) else lambda x: R(x).T
        )

    # -----------------------------------------------------
    # Arithmetic operations

    def __neg__(self: "Rotate") -> "Rotate":
        """Negate the rotation.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import coordinax as cx

        >>> Rz = jnp.asarray([[0, -1, 0], [1, 0,  0], [0, 0, 1]])
        >>> op = cx.ops.Rotate(Rz)
        >>> print((-op).R)
        [[ 0  1  0]
         [-1  0  0]
         [ 0  0 -1]]

        """
        R = (
            (self.R.func if isinstance(self.R, Neg) else Neg(self.R))
            if callable(self.R)
            else -self.R
        )
        return replace(self, R=R)

    def __matmul__(self: "Rotate", other: Any, /) -> Any:
        """Combine two Galilean rotations.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax as cx

        Two rotations can be combined:

        >>> theta1 = u.Q(45, "deg")
        >>> Rz1 = jnp.asarray([[jnp.cos(theta1), -jnp.sin(theta1), 0],
        ...                   [jnp.sin(theta1), jnp.cos(theta1),  0],
        ...                   [0,             0,              1]])
        >>> op1 = cx.ops.Rotate(Rz1)

        >>> theta2 = u.Q(90, "deg")
        >>> Rz2 = jnp.asarray([[jnp.cos(theta2), -jnp.sin(theta2), 0],
        ...                   [jnp.sin(theta2), jnp.cos(theta2),  0],
        ...                   [0,             0,              1]])
        >>> op2 = cx.ops.Rotate(Rz2)

        >>> op3 = op1 @ op2
        >>> op3
        Rotate(R=f32[3,3])

        >>> jnp.allclose(op3.R, op1.R @ op2.R)
        Array(True, dtype=bool)

        """
        return replace(self, R=self.R @ other.R)


@Rotate.from_.dispatch
def from_(obj: Rotate, /) -> Rotate:
    """Construct a Rotate from another Rotate.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax.ops as cxo

    >>> R = cxo.Rotate(jnp.eye(3))
    >>> cxo.Rotate.from_(R) is R
    True

    """
    return obj


@Rotate.from_.dispatch
def from_(obj: Callable, /) -> Rotate:
    """Construct a Rotate from a callable.

    The callable must have a return type annotation with shape ending in NxN
    (a square matrix).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.ops as cxo
    >>> from jaxtyping import Array, Real

    >>> def R_func(t) -> Real[Array, "3 3"]:
    ...     return jnp.eye(3)

    >>> R = cxo.Rotate.from_(R_func)
    >>> R
    Rotate(R=<function R_func at ...>)

    """
    # Validate return type has square matrix shape
    return_type = get_type_hints(obj, include_extras=True).get("return")
    if return_type is None:
        msg = "Callable must have a return type annotation."
        raise ValueError(msg)

    if not hasattr(return_type, "dims"):
        msg = "Callable return type must have jaxtyping shape annotation."
        raise ValueError(msg)

    dims = return_type.dims

    if not isinstance(dims, tuple):
        msg = "Callable return type dims must be a tuple."
        raise ValueError(msg)

    if len(dims) < 2:
        msg = f"Callable return type must have matrix shape (...,NxN), got {dims}"
        raise ValueError(msg)

    # Check if last two dimensions are equal (NxN)
    dim1, dim2 = dims[-2].size, dims[-1].size
    # Both should be the same (either literal numbers or same variable)
    if dim1 != dim2:
        msg = (
            "Callable return type must have square matrix shape (NxN), "
            f"got {dim1} x {dim2}"
        )
        raise ValueError(msg)

    return Rotate(obj)


@Rotate.from_.dispatch
def from_(obj: u.AbstractQuantity, /) -> Rotate:
    """Construct a Rotate from a Quantity.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    >>> cxo.Rotate.from_(u.Q(jnp.eye(3), ""))
    Rotate(rotation=i32[3,3])

    """
    return Rotate(u.ustrip("", obj))


@Rotate.from_.dispatch
def from_(obj: ArrayLike, /) -> Rotate:
    """Construct a Rotate from an Array.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.ops as cxo

    >>> cxo.Rotate.from_(jnp.eye(3))
    Rotate(rotation=i32[3,3])

    """
    return Rotate(jnp.asarray(obj))


# ============================================================================
# Simplification


@plum.dispatch
def simplify(op: Rotate, /, **kw: Any) -> AbstractOperator:
    """Simplify the Galilean rotation operator.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    An operator with a non-identity rotation matrix is not simplified:

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0,  0], [0, 0, 1]])
    >>> op = cx.ops.Rotate(Rz)
    >>> cx.ops.simplify(op)
    Rotate(rotation=i32[3,3])

    An operator with an identity rotation matrix is simplified:

    >>> op = cx.ops.Rotate(jnp.eye(3))
    >>> cx.ops.simplify(op)
    Identity()

    When two rotations are combined that cancel each other out, the result
    simplifies to an :class:`coordinax.ops.Identity`:

    >>> op = (  cx.ops.Rotate.from_euler("z", u.Q(45, "deg"))
    ...       @ cx.ops.Rotate.from_euler("z", u.Q(-45, "deg")))
    >>> cx.ops.simplify(op)
    Identity()

    """
    if not callable(op.R) and jnp.allclose(op.R, jnp.eye(3), **kw):
        return Identity()  # type: ignore[no-untyped-call]
    return op


# ============================================================================
# apply_op for Rotate on Quantity


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    x: ArrayLike,
    /,
    at: CsDict | None = None,
    usys: u.AbstractUnitSystem | None = None,
    **kw: Any,
) -> Array:
    """Apply Rotate to an Array(like) object.

    The Array is interpreted as equivalent to the data for a
    {class}`~coordinax.Vector` with a Cartesian chart (e.g.
    {class}`~coordinax.charts.Cartesian3D`) and {class}`~coordinax.roles.Point`
    role.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> op = cxo.Rotate(Rz)
    >>> q = u.Q([1, 0, 0], "km")
    >>> cxo.apply_op(op, None, q)
    Quantity['length'](Array([0, 1, 0], dtype=int32), unit='km')

    """
    del at, usys, kw  # Does not require an anchoring base-point.
    op_eval = eval_op(op, tau)
    return vec_matmul(op_eval.R, jnp.asarray(x))


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    x: u.AbstractQuantity,
    /,
    *,
    at: CsDict | None = None,
    usys: u.AbstractUnitSystem | None = None,
    **kw: Any,
) -> u.AbstractQuantity:
    """Apply Rotate to a Quantity.

    The Quantity is interpreted as equivalent to the data for a
    {class}`~coordinax.Vector` with a Cartesian chart (e.g.
    {class}`~coordinax.charts.Cartesian3D`) and {class}`~coordinax.roles.Point`
    role.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> op = cxo.Rotate(Rz)
    >>> q = u.Q([1, 0, 0], "km")
    >>> cxo.apply_op(op, None, q)
    Quantity['length'](Array([0, 1, 0], dtype=int32), unit='km')

    """
    del at, usys, kw  # Does not require an anchoring base-point.
    op_eval = eval_op(op, tau)
    return vec_matmul(op_eval.R, x)


_MSG_R_SHAPE: Final = "Rotate requires a square rotation matrix; got shape={shape!r}."
_MSG_R_X_SHAPE_MISMATCH: Final = (
    "Rotate(Point, chart=...) requires the chart's canonical Cartesian chart "
    "to have dimension matching the rotation matrix. "
    "Got R.shape={R.shape} and cartesian_chart={type(cart).__name__} "
    "with ndim={getattr(cart, 'ndim', None)!r} and components={comps!r}."
)


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.Point,
    chart: cxc.AbstractChart,
    x: CsDict,
    /,
    *,
    at: CsDict | None = None,
    usys: u.AbstractUnitSystem | None = None,
    **kw: Any,
) -> CsDict:
    """Apply a spatial rotation to a Point-valued coordinate dictionary.

    The point is rotated by converting to the chart's canonical Cartesian chart,
    applying the rotation in Cartesian components, then converting back.

    Notes
    -----
    - This dispatch is for non-product charts; Cartesian-product charts have a
      separate dispatch that rotates only matching spatial factors.
    - The rotation matrix must be square and its dimension must match the
      canonical Cartesian chart dimension.
    - Units are handled by packing Cartesian components into a common unit before
      rotation and restoring that unit afterward.

    """
    del role, at, kw  # Does not require an anchoring base-point.

    op_eval = eval_op(op, tau)
    R = op_eval.R

    cart = api.cartesian_chart(chart)
    comps = cart.components
    n = R.shape[0]

    if cart.ndim != n or len(comps) != n:
        msg = _MSG_R_X_SHAPE_MISMATCH.format(R=R, cart=cart, comps=comps)
        raise NotImplementedError(msg)

    # Convert point to canonical Cartesian chart.
    p_cart = api.point_transform(cart, chart, x, usys=usys)

    # Pack -> rotate -> unpack (batch-safe), preserving a shared unit.
    v, unit = _pack_uniform_unit(p_cart, keys=comps)  # (..., n)
    v_rot = jnp.einsum("ij,...j->...i", R, v)  # (..., n)
    p_cart_rot = _unpack_with_unit(v_rot, unit, comps)

    # Convert back to original chart.
    return api.point_transform(chart, cart, p_cart_rot, usys=usys)


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.Point,
    chart: cxc.AbstractCartesianProductChart,
    x: CsDict,
    /,
    *,
    at: CsDict | None = None,
    usys: u.AbstractUnitSystem | None = None,
    **__: Any,
) -> CsDict:
    """Apply a spatial rotation to a Point-valued coordinate dictionary.

    For Cartesian-product charts, this applies the rotation factorwise: each
    factor is rotated in its canonical Cartesian chart *iff* that Cartesian
    chart's dimension matches the rotation matrix. Factors that do not match
    (e.g. Time1D) are left unchanged.

    Notes
    -----
    - Points do not require an anchoring base-point.
    - The rotation matrix must be square.
    - Units are handled by packing Cartesian components into a shared unit before
      rotation and restoring that unit afterward.

    """
    op_eval = eval_op(op, tau)
    n = op_eval.R.shape[-1]

    parts = chart.split_components(x)

    def _maybe_rotate_factor(factor_chart: cxc.AbstractChart, part: CsDict) -> CsDict:
        cart = api.cartesian_chart(factor_chart)
        if cart.ndim != n or len(cart.components) != n:
            return part
        return api.apply_op(op, tau, role, factor_chart, part, at=at, usys=usys)

    rotated_parts = tuple(
        _maybe_rotate_factor(f, p) for f, p in zip(chart.factors, parts, strict=True)
    )
    return chart.merge_components(rotated_parts)


_MSG_R_NEEDS_AT: Final = (
    "Rotate({role}, chart={chart}) requires `at=` (the base point) when chart "
    "conversion is needed. Provide `at` as a Point-valued components dictionary in the "
    "same `chart` as `x`."
)

_MSG_R_VEL_NEEDS_AT: Final = (
    "Rotate(Vel, ...) requires `at=` (the base point) because the correct law "
    "depends on position for time-dependent rotations."
)

_MSG_R_ACC_NEEDS_AT_AND_VEL: Final = (
    "Rotate(Acc, ...) requires `at=` (base point) and `vel=` (the velocity at the same "
    "base point) because the correct law for time-dependent rotations depends on x "
    "and v."
)


def _validate_square(R: Any, /) -> int:
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(_MSG_R_SHAPE.format(shape=getattr(R, "shape", None)))
    return int(R.shape[0])


def _dR_dt(op: Rotate, tau: Any):
    """Time derivative of the materialized rotation matrix.

    Normative convention:
    - If `op.R` is a constant matrix, derivative is 0.
    - If `op.R` is callable, we interpret it as R(tau_numeric) and differentiate
      w.r.t. the (already-stripped) tau argument.
    """
    if callable(op.R):
        # If tau is a Quantity, treat the callable as expecting stripped tau.
        tau0 = u.ustrip("", tau) if isinstance(tau, u.AbstractQuantity) else tau
        return jax.jacfwd(op.R)(tau0)
    # constant matrix: d/dt = 0
    R = op.R
    return jnp.zeros_like(R)


def _d2R_dt2(op: Rotate, tau: Any):
    if callable(op.R):
        tau0 = u.ustrip("", tau) if isinstance(tau, u.AbstractQuantity) else tau
        return jax.jacfwd(jax.jacfwd(op.R))(tau0)
    R = op.R
    return jnp.zeros_like(R)


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.Pos,
    chart: cxc.AbstractChart,
    x: CsDict,
    /,
    *,
    at: CsDict | None = None,
    usys: u.AbstractUnitSystem | None = None,
    **__: Any,
) -> CsDict:
    """Apply Rotate to a Pos-valued CsDict.

    For non-Cartesian charts, we must convert via physical tangent transforms,
    which depend on the base point `at`.
    """
    del role
    op_eval = eval_op(op, tau)
    R = op_eval.R
    n = _validate_square(R)

    cart = api.cartesian_chart(chart)
    comps_cart = cart.components
    if cart.ndim != n or len(comps_cart) != n:
        msg = (
            f"Rotate(Pos, chart=...) requires cartesian_chart(chart) to match R. "
            f"Got R.shape={R.shape}, cart={type(cart).__name__}, ndim={cart.ndim!r}, "
            f"components={comps_cart!r}."
        )
        raise NotImplementedError(msg)

    # If the chart basis depends on position (i.e. chart != cart), we need `at`.
    if chart != cart:
        x = eqx.error_if(
            x,
            at is None,
            _MSG_R_NEEDS_AT.format(role="Pos", chart=type(chart).__name__),
        )
    at0 = at  # may be None only for charts whose frame ignores `at` (Cartesian)

    # Rotate the base point to express output components in the chart frame at p'.
    at_rot = None
    if at0 is not None:
        at_rot = api.apply_op(op, tau, cxr.point, chart, at0, usys=usys)

    # Pack chart components -> ambient Cartesian components via the orthonormal
    # frame at p.
    keys_chart = chart.components
    v_chart, unit = _pack_uniform_unit(x, keys=keys_chart)  # (..., n)

    B = api.frame_cart(
        chart, at=at0 if at0 is not None else {}, usys=usys
    )  # (..., N, n)
    v_cart = cxf.pushforward(B, v_chart)  # (..., N)

    # Rotate in ambient Cartesian components.
    v_cart_rot = jnp.einsum("ij,...j->...i", R, v_cart)  # (..., N)

    # Pull back into chart physical components at p'.
    B_rot = api.frame_cart(chart, at=at_rot if at_rot is not None else {}, usys=usys)
    g = api.metric_of(chart)
    v_chart_rot = cxf.pullback(g, B_rot, v_cart_rot)  # (..., n)

    return _unpack_with_unit(v_chart_rot, unit, keys_chart)


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.Pos,
    chart: cxc.AbstractCartesianProductChart,
    x: CsDict,
    /,
    *,
    at: CsDict | None = None,
    usys: u.AbstractUnitSystem | None = None,
    **kw: Any,
) -> CsDict:
    """Apply Rotate to Pos on a Cartesian-product chart, factorwise.

    Rotates only those factors whose canonical Cartesian chart dimension matches
    `R`; leaves other factors unchanged.
    """
    del role
    op_eval = eval_op(op, tau)
    n = _validate_square(op_eval.R)

    parts = chart.split_components(x)
    at_parts = chart.split_components(at) if at is not None else None

    def _maybe(
        factor_chart: cxc.AbstractChart, part: CsDict, at_part: CsDict | None
    ) -> CsDict:
        cart = api.cartesian_chart(factor_chart)
        if cart.ndim != n or len(cart.components) != n:
            return part
        # Re-dispatch to the non-product Pos rule for the factor.
        return api.apply_op(
            op, tau, cxr.pos, factor_chart, part, at=at_part, usys=usys, **kw
        )

    rotated_parts = tuple(
        _maybe(f, p, (None if at_parts is None else at_parts[i]))
        for i, (f, p) in enumerate(zip(chart.factors, parts, strict=True))
    )
    return chart.merge_components(rotated_parts)


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.Vel,
    chart: cxc.AbstractChart,
    x: CsDict,
    /,
    *,
    at: CsDict | None = None,
    usys: u.AbstractUnitSystem | None = None,
    **__: Any,
) -> CsDict:
    r"""Apply Rotate to a Vel-valued CsDict.

    Correct Euclidean law (in Cartesian components):
    \[
        v'(\tau) = R(\tau)\,v(\tau) + \dot R(\tau)\,x(\tau).
    \]

    Therefore this dispatch requires `at=` (the base point coordinates) unless
    \dot R is identically zero (constant rotation) AND no chart conversion is needed.
    """
    del role
    op_eval = eval_op(op, tau)
    R = op_eval.R
    n = _validate_square(R)

    cart = api.cartesian_chart(chart)
    comps_cart = cart.components
    if cart.ndim != n or len(comps_cart) != n:
        msg = (
            f"Rotate(Vel, chart=...) requires cartesian_chart(chart) to match R. "
            f"Got R.shape={R.shape}, cart={type(cart).__name__}, ndim={cart.ndim!r}, "
            f"components={comps_cart!r}."
        )
        raise NotImplementedError(msg)

    # Time dependence introduces + dR/dt * x, which requires the base point.
    dR = _dR_dt(op, tau)
    needs_at_for_dR = not jnp.allclose(dR, 0)

    if chart != cart or needs_at_for_dR:
        x = eqx.error_if(x, at is None, _MSG_R_VEL_NEEDS_AT)
    at0 = at  # required if chart!=cart or time-dependent

    # Rotate the base point to express output components in the chart frame at p'.
    at_rot = None
    if at0 is not None:
        at_rot = api.apply_op(op, tau, cxr.point, chart, at0, usys=usys)

    # Pack velocity chart components -> ambient Cartesian via frame at p.
    keys_chart = chart.components
    v_chart, v_unit = _pack_uniform_unit(x, keys=keys_chart)  # (..., n)

    B = api.frame_cart(chart, at=at0 if at0 is not None else {}, usys=usys)
    v_cart = cxf.pushforward(B, v_chart)  # (..., N)

    # Rotate the R v term.
    v_cart_rot = jnp.einsum("ij,...j->...i", R, v_cart)

    # Add + dR/dt * x for time-dependent rotations.
    if needs_at_for_dR:
        x_cart_dict = api.point_transform(cart, chart, at0, usys=usys)  # type: ignore[arg-type]
        x_cart, _x_unit = _pack_uniform_unit(x_cart_dict, keys=comps_cart)  # (..., n)
        v_cart_rot = v_cart_rot + jnp.einsum("ij,...j->...i", dR, x_cart)

    # Pull back into chart velocity components at p'.
    B_rot = api.frame_cart(chart, at=at_rot if at_rot is not None else {}, usys=usys)
    g = api.metric_of(chart)
    v_chart_rot = cxf.pullback(g, B_rot, v_cart_rot)  # (..., n)

    return _unpack_with_unit(v_chart_rot, v_unit, keys_chart)


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.Vel,
    chart: cxc.AbstractCartesianProductChart,
    x: CsDict,
    /,
    *,
    at: CsDict | None = None,
    usys: u.AbstractUnitSystem | None = None,
    **kw: Any,
) -> CsDict:
    """Apply Rotate to Vel on a Cartesian-product chart, factorwise."""
    del role
    op_eval = eval_op(op, tau)
    n = _validate_square(op_eval.R)

    parts = chart.split_components(x)
    at_parts = chart.split_components(at) if at is not None else None

    def _maybe(chart: cxc.AbstractChart, part: CsDict, at: CsDict | None) -> CsDict:
        cart = api.cartesian_chart(chart)
        if cart.ndim != n or len(cart.components) != n:
            return part
        return api.apply_op(op, tau, cxr.vel, chart, part, at=at, usys=usys, **kw)

    rotated_parts = tuple(
        _maybe(f, p, (None if at_parts is None else at_parts[i]))
        for i, (f, p) in enumerate(zip(chart.factors, parts, strict=True))
    )
    return chart.merge_components(rotated_parts)


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.Acc,
    chart: cxc.AbstractChart,
    x: CsDict,
    /,
    *,
    at: CsDict | None = None,
    vel: CsDict | None = None,
    usys: u.AbstractUnitSystem | None = None,
    **__: Any,
) -> CsDict:
    r"""Apply Rotate to an Acc-valued CsDict (physical acceleration).

    In ambient Cartesian components (Euclidean metric), the correct law is:
    \[
        a'(\tau) = R(\tau)\,a(\tau) + 2\dot R(\tau)\,v(\tau) + \ddot R(\tau)\,x(\tau).
    \]

    Therefore this dispatch requires:
    - `at=` whenever chart conversion is needed OR \dot R/\ddot R is nonzero,
    - `vel=` whenever \dot R is nonzero (Coriolis-like term) OR chart conversion
      is needed to interpret v in the correct physical frame.
    """
    del role
    op_eval = eval_op(op, tau)
    R = op_eval.R
    n = _validate_square(R)

    cart = api.cartesian_chart(chart)
    comps_cart = cart.components
    if cart.ndim != n or len(comps_cart) != n:
        msg = (
            f"Rotate(Acc, chart=...) requires cartesian_chart(chart) to match R. "
            f"Got R.shape={R.shape}, cart={type(cart).__name__}, ndim={cart.ndim!r}, "
            f"components={comps_cart!r}."
        )
        raise NotImplementedError(msg)

    dR = _dR_dt(op, tau)
    ddR = _d2R_dt2(op, tau)
    needs_at_for_ddR = not jnp.allclose(ddR, 0)
    needs_at_for_dR = not jnp.allclose(dR, 0)
    needs_at = (chart != cart) or needs_at_for_dR or needs_at_for_ddR
    needs_vel = (chart != cart) or needs_at_for_dR  # v-term appears if dR != 0

    x = eqx.error_if(x, needs_at and at is None, _MSG_R_ACC_NEEDS_AT_AND_VEL)
    x = eqx.error_if(x, needs_vel and vel is None, _MSG_R_ACC_NEEDS_AT_AND_VEL)

    at0 = at  # required if needs_at
    vel0 = vel  # required if needs_vel

    # Rotate the base point to express output components in the chart frame at p'.
    at_rot = None
    if at0 is not None:
        at_rot = api.apply_op(op, tau, cxr.point, chart, at0, usys=usys)

    # Pack acceleration chart components -> ambient Cartesian via frame at p.
    keys_chart = chart.components
    a_chart, a_unit = _pack_uniform_unit(x, keys=keys_chart)  # (..., n)

    B = api.frame_cart(
        chart, at=at0 if at0 is not None else {}, usys=usys
    )  # (..., N, n)
    a_cart = cxf.pushforward(B, a_chart)  # (..., N)

    # Base term: R a
    a_cart_rot = jnp.einsum("ij,...j->...i", R, a_cart)

    # Add + 2 dR v term (requires vel)
    if needs_at_for_dR:
        # Convert vel (chart physical components at p) -> ambient Cartesian.
        v_chart, _v_unit = _pack_uniform_unit(vel0, keys=keys_chart)  # (..., n)
        v_cart = cxf.pushforward(B, v_chart)  # (..., N)
        a_cart_rot = a_cart_rot + 2.0 * jnp.einsum("ij,...j->...i", dR, v_cart)

    # Add + ddR x term (requires at)
    if needs_at_for_ddR:
        x_cart_dict = api.point_transform(cart, chart, at0, usys=usys)  # type: ignore[arg-type]
        x_cart, _x_unit = _pack_uniform_unit(x_cart_dict, keys=comps_cart)  # (..., n)
        a_cart_rot = a_cart_rot + jnp.einsum("ij,...j->...i", ddR, x_cart)

    # Pull back into chart acceleration components at p'.
    B_rot = api.frame_cart(chart, at=at_rot if at_rot is not None else {}, usys=usys)
    g = api.metric_of(chart)
    a_chart_rot = cxf.pullback(g, B_rot, a_cart_rot)  # (..., n)

    return _unpack_with_unit(a_chart_rot, a_unit, keys_chart)


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.Acc,
    chart: cxc.AbstractCartesianProductChart,
    x: CsDict,
    /,
    *,
    at: CsDict | None = None,
    vel: CsDict | None = None,
    usys: u.AbstractUnitSystem | None = None,
    **kw: Any,
) -> CsDict:
    """Apply Rotate to Acc on a Cartesian-product chart, factorwise.

    Rotates only those factors whose canonical Cartesian chart dimension matches `R`;
    leaves other factors unchanged. `at` and `vel` are partitioned factorwise and
    passed through to the factor re-dispatch.
    """
    del role
    op_eval = eval_op(op, tau)
    n = _validate_square(op_eval.R)

    parts = chart.split_components(x)
    at_parts = chart.split_components(at) if at is not None else None
    vel_parts = chart.split_components(vel) if vel is not None else None

    def _maybe(
        chart: cxc.AbstractChart, part: CsDict, at: CsDict | None, vel: CsDict | None
    ) -> CsDict:
        cart = api.cartesian_chart(chart)
        if cart.ndim != n or len(cart.components) != n:
            return part
        return api.apply_op(
            op, tau, cxr.acc, chart, part, at=at, vel=vel, usys=usys, **kw
        )

    rotated_parts = tuple(
        _maybe(
            f,
            p,
            (None if at_parts is None else at_parts[i]),
            (None if vel_parts is None else vel_parts[i]),
        )
        for i, (f, p) in enumerate(zip(chart.factors, parts, strict=True))
    )
    return chart.merge_components(rotated_parts)
