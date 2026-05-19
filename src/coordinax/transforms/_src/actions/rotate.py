"""Galilean coordinate transformations."""

__all__ = ("Rotate",)


from dataclasses import replace
from operator import itemgetter

from collections.abc import Callable
from jaxtyping import Array, Shaped
from typing import Any, Final, TypeAlias, cast, final, get_type_hints

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

import coordinax.api.transforms as cxfmapi
import coordinax.charts as cxc
import coordinax.representations as cxr
from .base import AbstractTransform, materialize_transform
from .custom_types import CDict, HasShape, OptUSys
from .identity import identity
from .utils import Neg
from coordinax.internal import pack_uniform_unit
from coordinax.transforms._src import groups

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
class Rotate(AbstractTransform):
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
    >>> import coordinax.main as cx
    >>> import coordinax.transforms as cxfm
    >>> import wadler_lindig as wl

    We can then create a rotation operator:

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0,  0], [0, 0, 1]])
    >>> op = cxfm.Rotate(Rz)
    >>> op
    Rotate(i64[3,3](jax))

    Rotation operators can be applied to {class}`~coordinax.Point` and other
    higher-level objects, with behavior depending on the role:

    >>> v = cx.Point.from_([1, 0, 0], "m")  # A cxr.Point vector
    >>> t = u.Q(1, "s")

    >>> print(op(t, v))  # equivalent to `cx.act(op, t, v)`
    <Point: chart=Cart3D (x, y, z) [m]
        [0 1 0]>

    This also works for a batch of vectors (as a note, it is more efficient to
    `jax.vmap` over the `jax.jit`ted operator):

    >>> v = cx.Point.from_([[1, 0, 0], [0, 1, 0]], "m")  # A Point vector
    >>> print(op(t, v))
    <Point: chart=Cart3D (x, y, z) [m]
        [[ 0  1  0]
         [-1  0  0]]>

    Rotations can also be applied to low-level coordinate dictionaries:

    >>> q = {"x": u.Q(1, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")}
    >>> nq = op(t, q)  # inferred chart & rep -> cxr.Point
    >>> wl.pprint(nq, short_arrays="compact", use_short_name=True)
    {'x': Q(0, unit='m'), 'y': Q(1, unit='m'), 'z': Q(0, unit='m')}

    In addition to the standard low-level objects, Rotation operators can be
    applied to {class}`~unxt.Quantity` and Array-like objects, taken to
    represent a Cartesian vectors. For Quantity, the role is inferred from the
    units, while Arrays are always points:

    >>> q = u.Q([1, 0, 0], "m")
    >>> t = u.Q(1, "s")
    >>> op(t, q)
    Q([0, 1, 0], 'm')

    This also works for a batch of vectors:

    >>> q = u.Q([[1, 0, 0], [0, 1, 0]], "m")
    >>> op(t, q)
    Q([[ 0,  1,  0],
       [-1,  0,  0]], 'm')

    You can make the rotation matrix time-dependent:

    >>> from jaxtyping import Array, Real
    >>> def R_func(t) -> Real[Array, "3 3"]:
    ...     theta = (jnp.pi / 4) * t.to_value("s")
    ...     st, ct = jnp.sin(theta), jnp.cos(theta)
    ...     return jnp.array([[ct, -st, 0], [st,  ct, 0], [0, 0, 1]])

    >>> R_op = cxfm.Rotate.from_(R_func)
    >>> R_op
    Rotate(<function R_func>)

    >>> t = u.Q(4, "s")  # R_func -> 180 degrees rotation
    >>> R_op(t, q).round(3)
    Q([[-1.,  0.,  0.],
       [-0., -1.,  0.]], 'm')

    """

    R: Shaped[Array, " N N"] | Callable[[Any], RMatrix]
    """The rotation vector."""

    @classmethod
    def groups(cls) -> frozenset[type]:
        """Return the groups to which this map belongs."""
        del cls
        return frozenset((groups.SpecialOrthogonalGroup, groups.DiffeomorphismGroup))

    def __init__(self, R: Any) -> None:
        object.__setattr__(self, "R", jnp.asarray(R) if not callable(R) else R)

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
        >>> import coordinax.main as cx

        >>> op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
        >>> op.R.round(2)
        Array([[ 0., -1.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  0.,  1.]], dtype=float64)

        """
        # JAX uses active (point-moving) rotation conventions; use directly.
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
        >>> import coordinax.main as cx

        >>> Rz = jnp.asarray([[0, -1, 0], [1, 0,  0], [0, 0, 1]])
        >>> op = cxfm.Rotate(Rz)
        >>> op.inverse
        Rotate(i64[3,3](jax))

        >>> jnp.allclose(op.R, op.inverse.R.T)
        Array(True, dtype=bool)

        """
        R = self.R
        return replace(  # TODO: a transposition wrapper
            self,
            R=jnp.swapaxes(R, -2, -1)
            if not callable(R)
            else lambda x: jnp.swapaxes(R(x), -2, -1),  # ty: ignore[call-top-callable]
        )

    # -----------------------------------------------------

    @staticmethod
    def _validate_square(R: HasShape, /) -> RMatrix:
        shape = R.shape
        return eqx.error_if(
            R, len(shape) != 2 or shape[0] != shape[1], _MSG_R_SHAPE.format(shape=shape)
        )

    @staticmethod
    def _validate_shape_match(
        R: HasShape, cart: cxc.AbstractChart[Any, Any, Any], /
    ) -> RMatrix:
        n = R.shape[0]
        return eqx.error_if(
            R,
            cart.ndim != n or len(cart.components) != n,
            _MSG_R_X_SHAPE_MISMATCH.format(R=R, cart=cart),
        )

    def _get_R(self, cart: cxc.AbstractChart[Any, Any, Any], /) -> RMatrix:
        R = self.R
        R = eqx.error_if(R, callable(R), "need to call `materialize_transform`.")
        R = self._validate_square(R)
        return self._validate_shape_match(R, cart)

    # -----------------------------------------------------
    # Arithmetic operations

    def __neg__(self: "Rotate") -> "Rotate":
        """Negate the rotation.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import coordinax.main as cx

        >>> Rz = jnp.asarray([[0, -1, 0], [1, 0,  0], [0, 0, 1]])
        >>> op = cxfm.Rotate(Rz)
        >>> print((-op).R)
        [[ 0  1  0]
         [-1  0  0]
         [ 0  0 -1]]

        """
        R = (
            (self.R.param if isinstance(self.R, Neg) else Neg(self.R))
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
        >>> import coordinax.main as cx

        Two rotations can be combined:

        >>> theta1 = u.Q(45, "deg")
        >>> Rz1 = jnp.asarray([[jnp.cos(theta1), -jnp.sin(theta1), 0],
        ...                   [jnp.sin(theta1), jnp.cos(theta1),  0],
        ...                   [0,             0,              1]])
        >>> op1 = cxfm.Rotate(Rz1)

        >>> theta2 = u.Q(90, "deg")
        >>> Rz2 = jnp.asarray([[jnp.cos(theta2), -jnp.sin(theta2), 0],
        ...                   [jnp.sin(theta2), jnp.cos(theta2),  0],
        ...                   [0,             0,              1]])
        >>> op2 = cxfm.Rotate(Rz2)

        >>> op3 = op1 @ op2
        >>> op3
        Rotate(Q(f64[3,3], ''))

        >>> jnp.allclose(op3.R, op2.R @ op1.R)
        Array(True, dtype=bool)

        """
        if not isinstance(other, Rotate):
            return NotImplemented
        if callable(self.R) or callable(other.R):
            msg = "@ is not yet implemented for Rotate with callable R."
            raise NotImplementedError(msg)
        return replace(self, R=other.R @ self.R)


# ============================================================================
# Constructors


@Rotate.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Rotate], obj: Rotate, /) -> Rotate:
    """Construct a Rotate from another Rotate.

    >>> import quaxed.numpy as jnp
    >>> import coordinax.transforms as cxfm
    >>> R = cxfm.Rotate(jnp.eye(3))
    >>> cxfm.Rotate.from_(R) is R
    True

    """
    return obj


@Rotate.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Rotate], obj: Callable[..., Any], /) -> Rotate:
    """Construct a Rotate from a callable.

    The callable must have a return type annotation with shape ending in NxN (a
    square matrix).

    >>> import jax.numpy as jnp
    >>> import coordinax.transforms as cxfm
    >>> from jaxtyping import Array, Real

    >>> def R_func(t) -> Real[Array, "3 3"]:
    ...     return jnp.eye(3)

    >>> R = cxfm.Rotate.from_(R_func)
    >>> R
    Rotate(<function R_func>)

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
        raise TypeError(msg)

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


@Rotate.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Rotate], obj: AbcQ, /) -> Rotate:
    """Construct a Rotate from a Quantity.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm
    >>> cxfm.Rotate.from_(u.Q(jnp.eye(3), ""))
    Rotate(f64[3,3](jax))

    """
    return cls(u.ustrip("", obj))


@Rotate.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Rotate], obj: ArrayLike, /) -> Rotate:
    """Construct a Rotate from an Array.

    >>> import jax.numpy as jnp
    >>> import coordinax.transforms as cxfm
    >>> cxfm.Rotate.from_(jnp.eye(3))
    Rotate(f64[3,3](jax))

    """
    return cls(jnp.asarray(obj))


@Rotate.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Rotate], obj: jtransform.Rotation, /) -> Rotate:
    """Initialize from a `jax.scipy.spatial.transform.Rotation`.

    >>> import jax.numpy as jnp
    >>> from jax.scipy.spatial.transform import Rotation
    >>> import coordinax.main as cx

    >>> R = Rotation.from_euler("z", 90, degrees=True)
    >>> op = cxfm.Rotate.from_(R)

    >>> jnp.allclose(op.R, R.as_matrix())
    Array(True, dtype=bool)

    """
    return cls(obj.as_matrix())


# ============================================================================
# Simplification


@plum.dispatch
def simplify(op: Rotate, /, **kw: Any) -> AbstractTransform:
    """Simplify the Galilean rotation operator.

    >>> import quaxed.numpy as jnp
    >>> import coordinax.main as cx

    An operator with a non-identity rotation matrix is not simplified:

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0,  0], [0, 0, 1]])
    >>> op = cxfm.Rotate(Rz)
    >>> cxfm.simplify(op)
    Rotate(i64[3,3](jax))

    An operator with an identity rotation matrix is simplified:

    >>> op = cxfm.Rotate(jnp.eye(3))
    >>> cxfm.simplify(op)
    Identity()

    When two rotations are combined that cancel each other out, the result
    simplifies to an {class}`coordinax.ops.Identity`:

    >>> op = (  cxfm.Rotate.from_euler("z", u.Q(45, "deg"))
    ...       @ cxfm.Rotate.from_euler("z", u.Q(-45, "deg")))
    >>> cxfm.simplify(op)
    Identity()

    """
    if not callable(op.R) and jnp.allclose(op.R, jnp.eye(3), **kw):
        return identity
    return op


# ============================================================================
# act

# -----------------------------------------------
# Special dispatches for Array.
# These are interpreted as Cartesian coordinates in a Euclidean metric


@plum.dispatch
def act(
    op: Rotate,
    tau: Any,
    x: ArrayLike,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> Array:
    """Apply Rotate to an Array(like) object."""
    del kw  # Does not require an anchoring base-point.

    # Process Value
    x_arr = jnp.asarray(x)
    chart = cxc.guess_chart(x_arr)  # ty: ignore[invalid-assignment]
    if chart != chart.cartesian:
        msg = "act for Rotate with ArrayLike x requires a Cartesian chart."
        raise ValueError(msg)
    if rep != cxr.point:
        msg = "act for Rotate with ArrayLike x requires a point representation."
        raise TypeError(msg)

    # Process rotation
    op_eval = materialize_transform(op, tau)
    R = op_eval._get_R(chart)

    return jnp.einsum("ij,...j->...i", R, x_arr)


# -----------------------------------------------
# Special dispatches for Quantity.
# These are interpreted as Cartesian coordinates in a Euclidean metric
# The role is inferred from the dimensions.

_MSG_NOT_CART: Final = (
    "act({op}, ..., Quantity) requires Cartesian components. "
    "chart {name} is not its cartesian_chart."
)


@plum.dispatch
def act(
    op: Rotate,
    tau: Any,
    x: AbcQ,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> AbcQ:
    """Apply Rotate to a PointGeometry-roled Quantity."""
    del rep, kw

    # Process Value
    cart = chart.cartesian
    if chart != cart:
        msg = _MSG_NOT_CART.format(op=type(op).__name__, name=type(chart).__name__)
        raise ValueError(msg)

    # Rotation matrix
    op_eval = materialize_transform(op, tau)
    R = op_eval._get_R(chart)
    return jnp.einsum("ij,...j->...i", R, x)  # ty: ignore[invalid-return-type]


# -----------------------------------------------
# On CDict


@plum.dispatch
def act(
    op: Rotate,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    *,
    usys: OptUSys = None,
    **kw: Any,
) -> CDict:
    # Redispatch to geom-specific dispatch.
    out = cxfmapi.act(op, tau, x, chart, rep.geom_kind, rep, usys=usys, **kw)
    return cast("CDict", out)


@plum.dispatch
def act(
    op: Rotate,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    geom: cxr.PointGeometry,
    rep: cxr.Representation,
    /,
    *,
    usys: OptUSys = None,
    **kw: Any,
) -> CDict:
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
    del geom, rep, kw  # Does not require an anchoring base-point.

    cart = chart.cartesian
    # print("CART", cart.__class__, chart.__class__)
    comps_cart = cart.components

    op_eval = materialize_transform(op, tau)
    R = op_eval._get_R(cart)

    # Convert point to canonical Cartesian chart.
    p_cart = cxc.pt_map(x, chart, cart, usys=usys)

    # Pack -> rotate -> unpack (batch-safe)
    v, unit = pack_uniform_unit(
        p_cart, keys=comps_cart
    )  # (..., n)  # ty: ignore[no-matching-overload]
    v_rot = jnp.einsum("ij,...j->...i", R, v)  # (..., n)
    p_cart_rot = cxc.cdict(v_rot, unit, comps_cart)

    # Convert back to original chart.
    out = cxc.pt_map(p_cart_rot, cart, chart, usys=usys)
    return cast("CDict", out)


@plum.dispatch
def act(
    op: Rotate,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    geom: cxr.TangentGeometry,
    rep: cxr.Representation,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
    **kw: Any,
) -> CDict:
    """Apply a spatial rotation to a TangentGeometry coordinate dictionary.

    Rotation acts on tangent vectors via the Jacobian pushforward, not as a
    direct coordinate substitution.  The algorithm is:

    1. Push ``x`` to the chart's canonical Cartesian chart via the Jacobian.
    2. Pack Cartesian components to a common unit.
    3. Apply ``R`` via ``einsum`` in a batch-safe way.
    4. Pull the result back to the original chart via the inverse Jacobian
       evaluated at the rotated base point.

    For Cartesian charts the Jacobian is the identity, so steps 1 and 4 are
    no-ops and ``at`` is not required.  For all other charts (e.g. spherical)
    ``at`` **must** be supplied: it is the base point (in the original chart)
    at which the Jacobian is evaluated.

    Examples
    --------
    Rotate a Cartesian velocity vector by +90 degrees about ``z``:

    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.transforms as cxfm

    >>> op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> x = {"x": u.Q(1, "m/s"), "y": u.Q(0, "m/s"), "z": u.Q(0, "m/s")}
    >>> out = cxfm.act(op, None, x, cxc.cart3d, cxr.tangent_geom, cxr.coord_vel)
    >>> jnp.stack([out[c].to_value("m/s") for c in ("x", "y", "z")]).round(3)
    Array([0., 1., 0.], dtype=float64)

    Rotate a spherical velocity at a given base point:

    >>> import jax.numpy as jnp
    >>> op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> x = {"r": u.Q(1, "m/s"), "theta": u.Q(0, "rad/s"), "phi": u.Q(0, "rad/s")}
    >>> at = {"r": u.Q(1, "m"), "theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0, "rad")}
    >>> out = cxfm.act(op, None, x, cxc.sph3d, cxr.tangent_geom, cxr.coord_vel, at=at)
    >>> round(float(out["r"].to_value("m/s")), 3)  # radial component preserved
    1.0

    """
    del geom, kw

    cart = chart.cartesian
    op_eval = materialize_transform(op, tau)
    R = op_eval._get_R(cart)

    if chart is cart:
        # Cartesian chart: Jacobian is the identity — simple linear map.
        p_cart = x
    else:
        # Non-Cartesian chart: push tangent forward via Jacobian.
        if at is None:
            msg = (
                "act(Rotate, ..., TangentGeometry) on a non-Cartesian chart "
                f"({chart!r}) requires 'at' (base point in chart coords) so "
                "the Jacobian pushforward can be evaluated."
            )
            raise TypeError(msg)
        at_cart = cxc.pt_map(at, chart, cart, usys=usys)
        p_cart = cxr.tangent_map(x, chart, rep, cart, at=at, usys=usys)  # ty: ignore[missing-argument]

    # Pack -> rotate -> unpack (batch-safe)
    comps_cart = cart.components
    v, unit = pack_uniform_unit(p_cart, keys=comps_cart)
    v_rot = jnp.einsum("ij,...j->...i", R, v)  # (..., n)
    p_cart_rot = cxc.cdict(v_rot, unit, comps_cart)

    if chart is cart:
        return p_cart_rot  # ty: ignore[invalid-return-type]

    # Rotate the base point in Cartesian to anchor the inverse Jacobian.
    at_cart_arr, at_unit = pack_uniform_unit(at_cart, keys=comps_cart)  # ty: ignore[no-matching-overload]
    at_cart_rot_arr = jnp.einsum("ij,...j->...i", R, at_cart_arr)
    at_cart_rot = cxc.cdict(at_cart_rot_arr, at_unit, comps_cart)

    # Pull rotated tangent back to original chart via inverse Jacobian.
    return cxr.tangent_map(p_cart_rot, cart, rep, chart, at=at_cart_rot, usys=usys)  # ty: ignore[missing-argument]


# -----------------------------------------------
# On CDict with Cartesian-product charts


@plum.dispatch
def act(
    op: Rotate,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractCartesianProductChart,
    geom: cxr.PointGeometry,
    rep: cxr.Representation,
    /,
    *,
    usys: OptUSys = None,
    **kw: Any,
) -> CDict:
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
    op_eval = materialize_transform(op, tau)
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
        factor_chart: cxc.AbstractChart[Any, Any, Any],
        part: CDict,
        /,
        **ats: CDict | None,
    ) -> CDict:
        # Determine if this factor's chart should be rotated.
        cart = factor_chart.cartesian
        if cart.ndim != n or len(cart.components) != n:
            return part

        # Apply rotation
        out = cxfmapi.act(op_eval, tau, part, factor_chart, geom, rep, usys=usys, **ats)
        return cast("CDict", out)

    rotated_parts = tuple(
        _maybe(f, p, **jtu.map(itemgetter(i), ats))
        for i, (f, p) in enumerate(zip(chart.factors, parts, strict=True))
    )
    return chart.merge_components(rotated_parts)
