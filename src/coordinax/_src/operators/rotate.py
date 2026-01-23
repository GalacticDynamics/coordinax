"""Galilean coordinate transformations."""
# ruff:noqa: N803

__all__ = ("Rotate",)


from dataclasses import replace
from operator import itemgetter

from collections.abc import Callable
from jaxtyping import Array, Shaped
from typing import Any, Final, TypeAlias, final, get_type_hints

import equinox as eqx
import jax
import jax.scipy.spatial.transform as jtransform
import jax.tree as jtu
import plum
from jax.typing import ArrayLike
from quax import quaxify

import quaxed.numpy as jnp
import unxt as u
from unxt import AbstractQuantity as AbcQ

from .base import AbstractOperator, Neg, eval_op
from coordinax._src import api, charts as cxc, roles as cxr, transformations as cxt
from coordinax._src.custom_types import CsDict, HasShape, OptUSys
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.identity import Identity
from coordinax._src.transformations.utils import pack_uniform_unit, unpack_with_unit

vec_matmul = quaxify(jax.numpy.vectorize(jax.numpy.matmul, signature="(N,N),(N)->(N)"))

RMatrix: TypeAlias = Shaped[Array, " N N"]

_MSG_R_SHAPE: Final = "Rotate requires a square rotation matrix; got shape={shape!r}."

_MSG_R_X_SHAPE_MISMATCH: Final = (
    "Rotate() requires the chart's canonical Cartesian chart "
    "to have dimension matching the rotation matrix. "
    "Got R.shape={R.shape} and cartesian_chart={cart.__class__.__name__} "
    "with ndim={cart.ndim!r}."
)


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
    >>> import wadler_lindig as wl

    We can then create a rotation operator:

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0,  0], [0, 0, 1]])
    >>> op = cx.ops.Rotate(Rz)
    >>> op
    Rotate(i64[3,3](jax))

    Rotation operators can be applied to {class}`~coordinax.Vector` and other
    higher-level objects, with behavior depending on the role:

    >>> v = cx.Vector.from_([1, 0, 0], "m")  # A cxr.Point vector
    >>> t = u.Q(1, "s")

    >>> print(op(t, v))  # equivalent to `cx.apply_op(op, t, v)`
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [0 1 0]>

    >>> v_pos = cx.as_pos(v)  # A cxr.PhysDisp vector
    >>> print(op(t, v_pos))
    <Vector: chart=Cart3D, role=Pos (x, y, z) [m]
        [0. 1. 0.]>

    >>> v = cx.Vector.from_([1, 0, 0], "m/s")  # A cxr.PhysVel vector
    >>> print(op(t, v))
    <Vector: chart=Cart3D, role=Vel (x, y, z) [m / s]
        [0. 1. 0.]>

    This also works for a batch of vectors (as a note, it is more efficient to
    `jax.vmap` over the `jax.jit`ted operator):

    >>> v = cx.Vector.from_([[1, 0, 0], [0, 1, 0]], "m")  # A cxr.Point vector
    >>> print(op(t, v))
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [[ 0  1  0]
         [-1  0  0]]>

    Rotations can also be applied to low-level coordinate dictionaries:

    >>> q = {"x": u.Q(1, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")}
    >>> nq = op(t, q)  # inferred chart & role -> cxr.Point
    >>> wl.pprint(nq, short_arrays="compact", use_short_name=True)
    {'x': Q(0, unit='m'), 'y': Q(1, unit='m'), 'z': Q(0, unit='m')}

    >>> q = {"x": u.Q(1, "m/s"), "y": u.Q(0, "m/s"), "z": u.Q(0, "m/s")}
    >>> nq = cx.ops.apply_op(op, t, cxr.phys_vel, cxc.cart3d, q)  # explicit role & chart
    >>> wl.pprint(nq, short_arrays="compact", use_short_name=True)
    {'x': Q(0., unit='m / s'), 'y': Q(1., unit='m / s'), 'z': Q(0., unit='m / s')}

    In addition to the standard low-level objects, Rotation operators can be
    applied to {class}`~unxt.Quantity` and Array-like objects, taken to
    represent a Cartesian vectors. For Quantity, the role is inferred from the
    units, for Arrays it is always {class}`~coordinax.roles.Point`:

    >>> q = u.Q([1, 0, 0], "m")
    >>> t = u.Q(1, "s")
    >>> op(t, q)
    Quantity(Array([0, 1, 0], dtype=int64), unit='m')

    This also works for a batch of vectors:

    >>> q = u.Q([[1, 0, 0], [0, 1, 0]], "m")
    >>> op(t, q)
    Quantity(Array([[ 0,  1,  0],
                    [-1,  0,  0]], dtype=int64), unit='m')

    You can make the rotation matrix time-dependent:

    >>> from jaxtyping import Array, Real
    >>> def R_func(t) -> Real[Array, "3 3"]:
    ...     theta = (jnp.pi / 4) * t.to_value("s")
    ...     st, ct = jnp.sin(theta), jnp.cos(theta)
    ...     return jnp.array([[ct, -st, 0], [st,  ct, 0], [0, 0, 1]])

    >>> R_op = cx.ops.Rotate.from_(R_func)
    >>> R_op
    Rotate(<function R_func>)

    >>> t = u.Q(4, "s")  # R_func -> 180 degrees rotation
    >>> R_op(t, q).round(3)
    Quantity(Array([[-1.,  0.,  0.],
                    [-0., -1.,  0.]], dtype=float64), unit='m')

    """

    R: Shaped[Array, " N N"] | Callable[[Any], RMatrix]
    """The rotation vector."""

    # -----------------------------------------------------
    # Constructors

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
        R = jtransform.Rotation.from_euler(
            seq, u.ustrip("deg", angles), degrees=True
        ).as_matrix()
        return cls(R)

    # -----------------------------------------------------

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
            self,
            R=jnp.swapaxes(R, -2, -1)
            if not callable(R)
            else lambda x: jnp.swapaxes(R(x), -2, -1),
        )

    # -----------------------------------------------------

    @staticmethod
    def _validate_square(R: HasShape, /) -> RMatrix:
        shape = R.shape
        R = eqx.error_if(
            R, len(shape) != 2 or shape[0] != shape[1], _MSG_R_SHAPE.format(shape=shape)
        )
        return R

    @staticmethod
    def _validate_shape_match(R: HasShape, cart: cxc.AbstractChart, /) -> RMatrix:
        n = R.shape[0]
        R = eqx.error_if(
            R,
            cart.ndim != n or len(cart.components) != n,
            _MSG_R_X_SHAPE_MISMATCH.format(R=R, cart=cart),
        )
        return R

    def _get_R(self, cart: cxc.AbstractChart, /) -> RMatrix:
        R = self.R
        R = eqx.error_if(R, callable(R), "need to call `eval_op`.")
        R = self._validate_square(R)
        R = self._validate_shape_match(R, cart)
        return R

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
        """Combine two Rotations.

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
        if not isinstance(other, Rotate):
            return NotImplemented
        return replace(self, R=self.R @ other.R)


# ============================================================================
# Constructors


@Rotate.from_.dispatch
def from_(cls: type[Rotate], obj: Rotate, /) -> Rotate:
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
def from_(cls: type[Rotate], obj: Callable, /) -> Rotate:
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

    return cls(obj)


@Rotate.from_.dispatch
def from_(cls: type[Rotate], obj: AbcQ, /) -> Rotate:
    """Construct a Rotate from a Quantity.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    >>> cxo.Rotate.from_(u.Q(jnp.eye(3), ""))
    Rotate(rotation=i32[3,3])

    """
    return cls(u.ustrip("", obj))


@Rotate.from_.dispatch
def from_(cls: type[Rotate], obj: ArrayLike, /) -> Rotate:
    """Construct a Rotate from an Array.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.ops as cxo

    >>> cxo.Rotate.from_(jnp.eye(3))
    Rotate(rotation=i32[3,3])

    """
    return cls(jnp.asarray(obj))


@Rotate.from_.dispatch
def from_(cls: type[Rotate], obj: jtransform.Rotation, /) -> Rotate:
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
# apply_op

# -----------------------------------------------
# Special dispatches for Array.
# These are interpreted as Cartesian coordinates in a Euclidean metric


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.AbstractRole,
    chart: cxc.AbstractChart,
    x: ArrayLike,
    /,
    **kw: Any,
) -> Array:
    """Apply Rotate to an Array(like) object."""
    del kw  # Does not require an anchoring base-point.

    # Process Value
    x_arr = jnp.asarray(x)
    chart = api.guess_chart(x_arr)
    if chart != chart.cartesian:
        msg = "apply_op for Rotate with ArrayLike x requires a Cartesian chart."
        raise ValueError(msg)
    if not isinstance(role, cxr.Point):
        msg = "apply_op for Rotate with ArrayLike x requires Point role."
        raise TypeError(msg)

    # Process rotation
    op_eval = eval_op(op, tau)
    R = op_eval._get_R(chart)

    return jnp.einsum("ij,...j->...i", R, x_arr)


# -----------------------------------------------
# Special dispatches for Quantity.
# These are interpreted as Cartesian coordinates in a Euclidean metric
# The role is inferred from the dimensions.

_MSG_NOT_CART: Final = (
    "apply_op({op}, ..., Quantity) requires Cartesian components. "
    "chart {name} is not its cartesian_chart."
)


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.Point,
    chart: cxc.AbstractChart,
    x: AbcQ,
    /,
    **kw: Any,
) -> AbcQ:
    """Apply Rotate to a Point-roled Quantity."""
    del role, kw

    # Process Value
    cart = chart.cartesian
    if chart != cart:
        msg = _MSG_NOT_CART.format(op=type(op).__name__, name=type(chart).__name__)
        raise ValueError(msg)

    # Rotation matrix
    op_eval = eval_op(op, tau)
    R = op_eval._get_R(chart)
    return jnp.einsum("ij,...j->...i", R, x)


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.PhysDisp,
    chart: cxc.AbstractChart,
    x: AbcQ,
    /,
    **kw: Any,
) -> AbcQ:
    """Apply Rotate to a Pos-roled Quantity.

    Interprets ``x`` as Cartesian physical displacement components in the
    chart inferred from its shape. Therefore the inferred chart must already be
    the canonical Cartesian chart.

    """
    del role, kw

    # Infer chart from shape; must be Cartesian.
    cart = chart.cartesian
    if chart != cart:
        raise ValueError(
            _MSG_NOT_CART.format(op=type(op).__name__, name=type(chart).__name__)
        )

    # Materialize and validate rotation against that Cartesian chart.
    op_eval = eval_op(op, tau)
    R = op_eval._get_R(cart)

    return jnp.einsum("ij,...j->...i", R, x)


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.PhysVel,
    chart: cxc.AbstractChart,
    x: AbcQ,
    /,
    **kw: Any,
) -> AbcQ:
    """Apply Rotate to a Vel-roled Quantity.

    Interprets ``x`` as Cartesian physical displacement components in the
    chart inferred from its shape. Therefore the inferred chart must already be
    the canonical Cartesian chart.

    """
    raise NotImplementedError("TODO")  # noqa: EM101


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.PhysAcc,
    chart: cxc.AbstractChart,
    x: AbcQ,
    /,
    **kw: Any,
) -> AbcQ:
    """Apply Rotate to a PhysAcc-roled Quantity.

    Interprets ``x`` as Cartesian physical displacement components in the
    chart inferred from its shape. Therefore the inferred chart must already be
    the canonical Cartesian chart.

    """
    raise NotImplementedError("TODO")  # noqa: EM101


# -----------------------------------------------
# On CsDict


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.Point,
    chart: cxc.AbstractChart,
    x: CsDict,
    /,
    usys: OptUSys = None,
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
    del role, kw  # Does not require an anchoring base-point.

    cart = chart.cartesian
    comps_cart = cart.components

    op_eval = eval_op(op, tau)
    R = op_eval._get_R(cart)

    # Convert point to canonical Cartesian chart.
    p_cart = api.point_transform(cart, chart, x, usys=usys)

    # Pack -> rotate -> unpack (batch-safe)
    v, unit = pack_uniform_unit(p_cart, keys=comps_cart)  # (..., n)
    v_rot = jnp.einsum("ij,...j->...i", R, v)  # (..., n)
    p_cart_rot = unpack_with_unit(v_rot, unit, comps_cart)

    # Convert back to original chart.
    return api.point_transform(chart, cart, p_cart_rot, usys=usys)


_MSG_NEEDS_AT: Final = (
    "Rotate(x) requires `at=` (the base point) when chart conversion is needed. "
    "Provide `at` as a Point-valued components dictionary in the same `chart` as `x`."
)

_MSG_NEEDS_AT_VEL: Final = (
    "Rotate(x) requires `at_vel=`, the velocity at the base point."
)


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.PhysDisp,
    chart: cxc.AbstractChart,
    x: CsDict,
    /,
    *,
    at: CsDict | None = None,
    usys: OptUSys = None,
    **kw: Any,
) -> CsDict:
    """Apply Rotate to a Pos-valued CsDict.

    For non-Cartesian charts, we must convert via physical tangent transforms,
    which depend on the base point `at`.
    """
    del role, kw

    # Process Value
    cart = chart.cartesian
    x = eqx.error_if(x, chart != cart and at is None, _MSG_NEEDS_AT)

    # Process Rotation
    op_eval = eval_op(op, tau)
    R = op_eval._get_R(cart)

    # Rotate base point to express output components in the chart frame at p'.
    # None only for charts whose frame ignores `at` (Cartesian)
    at0 = {} if at is None else at
    at_rot = (
        {} if at is None else api.apply_op(op, tau, cxr.point, chart, at=at, usys=usys)
    )

    # Pack chart components -> ambient Cartesian components via the orthonormal
    # frame at p.
    keys_chart = chart.components
    v_chart, unit = pack_uniform_unit(x, keys=keys_chart)  # (..., n)

    # B (..., N, n)
    B = api.frame_cart(chart, at=at0, usys=usys)
    v_cart = cxt.pushforward(B, v_chart)  # (..., N)

    # Rotate in ambient Cartesian components.
    v_cart_rot = jnp.einsum("ij,...j->...i", R, v_cart)  # (..., N)

    # Pull back into chart physical components at p'.
    B_rot = api.frame_cart(chart, at=at_rot, usys=usys)
    g = api.metric_of(chart)
    v_chart_rot = cxt.pullback(g, B_rot, v_cart_rot)  # (..., n)

    return unpack_with_unit(v_chart_rot, unit, keys_chart)


def _dR_dt(R: RMatrix | Callable, tau: Any, /) -> Array:
    """Time derivative of the materialized rotation matrix.

    convention:
    - If `op.R` is a constant matrix, derivative is 0.
    - If `op.R` is callable, we interpret it as R(tau_numeric) and differentiate
      w.r.t. the (already-stripped) tau argument.

    """
    # Evaluate jacobian or return constant matrix (d/dt = 0)
    return jax.jacfwd(R)(tau) if callable(R) else jnp.zeros_like(R)


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.PhysVel,
    chart: cxc.AbstractChart,
    x: CsDict,
    /,
    *,
    at: CsDict | None = None,
    usys: OptUSys = None,
    **kw: Any,
) -> CsDict:
    r"""Apply Rotate to a Vel-valued CsDict.

    Correct Euclidean law (in Cartesian components):
    $$
        v'(\tau) = R(\tau)\,v(\tau) + \dot R(\tau)\,x(\tau).
    $$

    Therefore this dispatch requires `at=` (the base point coordinates) unless
    \dot R is identically zero (constant rotation) AND no chart conversion is needed.
    """
    del role, kw

    cart = chart.cartesian
    tau = jnp.astype(tau, float)

    # Process Rotation
    op_eval = eval_op(op, tau)
    R = op_eval._get_R(cart)

    # Time dependence introduces + dR/dt * x, which requires the base point.
    dR = _dR_dt(op.R, tau)
    needs_at_for_dR = jnp.logical_not(jnp.allclose(dR, 0))
    needs_at = (chart != cart) or needs_at_for_dR
    x = eqx.error_if(x, needs_at and at is None, _MSG_NEEDS_AT)

    # Rotate base point to express output components in the chart frame at p'.
    # required if chart!=cart or time-dependent
    # The only reason at0 -> 0 is safe is because we've already checked `x`!
    at0 = {k: jnp.zeros(()) for k in chart.components} if at is None else at
    at_rot = (
        {k: jnp.zeros(()) for k in chart.components}
        if at is None
        else api.apply_op(op, tau, cxr.point, chart, at=at, usys=usys)
    )

    # Pack velocity chart components -> ambient Cartesian via frame at p.
    keys_chart = chart.components
    v_chart, v_unit = pack_uniform_unit(x, keys=keys_chart)  # (..., n)

    B = api.frame_cart(chart, at=at0, usys=usys)
    v_cart = cxt.pushforward(B, v_chart)  # (..., N)

    # Rotate the R v term.
    v_cart_rot = jnp.einsum("ij,...j->...i", R, v_cart)

    # Add + dR/dt * x for time-dependent rotations.
    x_cart_dict = api.point_transform(cart, chart, at0, usys=usys)
    x_cart, _ = pack_uniform_unit(x_cart_dict, keys=cart.components)  # (..., n)
    v_cart_rot = jax.lax.select(
        needs_at_for_dR,
        v_cart_rot + jnp.einsum("ij,...j->...i", dR, x_cart),
        v_cart_rot,
    )

    # Pull back into chart velocity components at p'.
    B_rot = api.frame_cart(chart, at=at_rot, usys=usys)
    g = api.metric_of(chart)
    v_chart_rot = cxt.pullback(g, B_rot, v_cart_rot)  # (..., n)

    return unpack_with_unit(v_chart_rot, v_unit, keys_chart)


def _d2R_dt2(R: RMatrix | Callable[..., RMatrix], tau: Any, /) -> Array:
    return jax.jacfwd(jax.jacfwd(R))(tau) if callable(R) else jnp.zeros_like(R)


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.PhysAcc,
    chart: cxc.AbstractChart,
    x: CsDict,
    /,
    *,
    at: CsDict | None = None,
    at_vel: CsDict | None = None,
    usys: OptUSys = None,
    **kw: Any,
) -> CsDict:
    r"""Apply Rotate to an PhysAcc-valued CsDict (physical acceleration).

    In ambient Cartesian components (Euclidean metric), the correct law is:
    $$
        a'(\tau) = R(\tau)\,a(\tau) + 2\dot R(\tau)\,v(\tau) + \ddot R(\tau)\,x(\tau).
    $$

    Therefore this dispatch requires:

    - `at=` whenever chart conversion is needed OR \dot R/\ddot R is nonzero,
    - `at_vel=` whenever \dot R is nonzero (Coriolis-like term) OR chart
      conversion is needed to interpret v in the correct physical frame.
    """
    del role, kw

    cart = chart.cartesian
    comps_cart = cart.components

    op_eval = eval_op(op, tau)
    R = op_eval._get_R(cart)

    # Time dependence introduces + dR/dt * x, which requires the base point.
    dR = _dR_dt(op.R, tau)
    ddR = _d2R_dt2(op.R, tau)

    needs_at_for_dR = jnp.logical_not(jnp.allclose(dR, 0))
    needs_at_for_ddR = jnp.logical_not(jnp.allclose(ddR, 0))
    needs_at = (chart != cart) or needs_at_for_dR or needs_at_for_ddR
    x = eqx.error_if(x, needs_at and at is None, _MSG_NEEDS_AT)

    needs_vel = (chart != cart) or needs_at_for_dR  # v-term appears if dR != 0
    x = eqx.error_if(x, needs_vel and at_vel is None, _MSG_NEEDS_AT_VEL)

    # Rotate base point to express output components in the chart frame at p'.
    # required if chart!=cart or time-dependent
    at0 = {} if at is None else at
    vel0 = {} if at_vel is None else at_vel
    at_rot = (
        {} if at is None else api.apply_op(op, tau, cxr.point, chart, at=at, usys=usys)
    )

    # Pack acceleration chart components -> ambient Cartesian via frame at p.
    keys_chart = chart.components
    a_chart, a_unit = pack_uniform_unit(x, keys=keys_chart)  # (..., n)

    B = api.frame_cart(chart, at=at0, usys=usys)
    a_cart = cxt.pushforward(B, a_chart)  # (..., N)

    # Base term: R a
    a_cart_rot = jnp.einsum("ij,...j->...i", R, a_cart)

    # Add + 2 dR v term (requires vel)
    # Convert vel (chart physical components at p) -> ambient Cartesian.
    v_chart, _v_unit = pack_uniform_unit(vel0, keys=keys_chart)  # (..., n)
    v_cart = cxt.pushforward(B, v_chart)  # (..., N)
    a_cart_rot = jax.lax.select(
        needs_at_for_dR,
        a_cart_rot + 2.0 * jnp.einsum("ij,...j->...i", dR, v_cart),
        a_cart_rot,
    )

    # Add + ddR x term (requires at)
    x_cart_dict = api.point_transform(cart, chart, at0, usys=usys)
    x_cart, _x_unit = pack_uniform_unit(x_cart_dict, keys=comps_cart)  # (..., n)
    a_cart_rot = jax.lax.select(
        needs_at_for_ddR,
        a_cart_rot + jnp.einsum("ij,...j->...i", ddR, x_cart),
        a_cart_rot,
    )

    # Pull back into chart acceleration components at p'.
    B_rot = api.frame_cart(chart, at=at_rot, usys=usys)
    g = api.metric_of(chart)
    a_chart_rot = cxt.pullback(g, B_rot, a_cart_rot)  # (..., n)

    return unpack_with_unit(a_chart_rot, a_unit, keys_chart)


# -----------------------------------------------
# On CsDict with Cartesian-product charts


@plum.dispatch
def apply_op(
    op: Rotate,
    tau: Any,
    role: cxr.AbstractRole,
    chart: cxc.AbstractCartesianProductChart,
    x: CsDict,
    /,
    *,
    usys: OptUSys = None,
    **kw: Any,
) -> CsDict:
    """Apply a spatial rotation to a coordinate dictionary.

    For Cartesian-product charts, this applies the rotation factorwise: each
    factor is rotated in its canonical Cartesian chart *iff* that Cartesian
    chart's dimension matches the rotation matrix. Factors that do not match
    (e.g. Time1D) are left unchanged.

    Notes
    -----
    - The rotation matrix must be square.
    - Units are handled by packing Cartesian components into a shared unit
      before rotation and restoring that unit afterward.
    - Kwarg requirements depend on role; eg. Points do not require an anchoring
      base-point.

    """
    op_eval = eval_op(op, tau)
    n = op_eval._validate_square(op_eval.R).shape[-1]

    # Factorize the inputs
    n_factors = len(chart.factors)
    parts = chart.split_components(x)
    ats = {
        k: chart.split_components(v) if v is not None else [None] * n_factors
        for k, v in kw.items()
        if k.startswith("at")
    }

    def _maybe(
        factor_chart: cxc.AbstractChart, part: CsDict, /, **ats: CsDict | None
    ) -> CsDict:
        # Determine if this factor's chart should be rotated.
        cart = factor_chart.cartesian
        if cart.ndim != n or len(cart.components) != n:
            return part

        # Apply rotation
        return api.apply_op(op_eval, tau, role, factor_chart, part, usys=usys, **ats)

    rotated_parts = tuple(
        _maybe(f, p, **jtu.map(itemgetter(i), ats))
        for i, (f, p) in enumerate(zip(chart.factors, parts, strict=True))
    )
    return chart.merge_components(rotated_parts)
