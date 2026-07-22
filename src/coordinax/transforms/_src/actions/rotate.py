"""Galilean coordinate transformations."""

__all__ = ("Rotate",)


from dataclasses import replace

from collections.abc import Callable
from jaxtyping import Array, Shaped
from typing import Any, TypeAlias, cast, final, get_type_hints

import jax
import jax.scipy.spatial.transform as jtransform
import plum
from jax.typing import ArrayLike

import quaxed.numpy as jnp
import unxt as u
from unxt import AbstractQuantity as AbcQ

import coordinax.charts as cxc
import coordinax.representations as cxr
from .base import AbstractTransform
from .custom_types import CDict, OptUSys
from .identity import identity
from .linear import AbstractLinearTransform
from .prolong import _attach_leaf, _strip_leaf, _tau_value_unit, prolong_slot
from .utils import Neg
from coordinax.internal import pack_uniform_unit
from coordinax.transforms._src import groups

RMatrix: TypeAlias = Shaped[Array, " N N"]


@final
class Rotate(AbstractLinearTransform):
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
    `jax.vmap` over the `jax.jit`-ed operator):

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
        >>> import coordinax as cx

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
        >>> import coordinax as cx

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

    @property
    def _raw_matrix(self) -> Any:
        return self.R

    # -----------------------------------------------------
    # Arithmetic operations

    def __neg__(self: "Rotate") -> "Rotate":
        """Negate the rotation.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import coordinax as cx

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
        >>> import coordinax as cx

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
    >>> import coordinax as cx

    >>> R = Rotation.from_euler("z", 90, degrees=True)
    >>> op = cxfm.Rotate.from_(R)

    >>> jnp.allclose(op.R, R.as_matrix())
    Array(True, dtype=bool)

    """
    return cls(obj.as_matrix())


# ============================================================================
# Simplification


@plum.dispatch
def simplify(op: Rotate, /, *, approx: bool = True, **kw: Any) -> AbstractTransform:
    """Simplify the Galilean rotation operator.

    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

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
    if approx and not callable(op.R) and jnp.allclose(op.R, jnp.eye(3), **kw):
        return identity
    return op


@plum.dispatch
def _merge(a: Rotate, b: Rotate, /) -> AbstractTransform | None:
    """Merge two adjacent rotations (``a`` applied first) into one.

    Static rotations combine as ``a @ b``; a time-dependent (callable) matrix on
    either side is left un-merged.
    """
    if callable(a.R) or callable(b.R):
        return None
    return a @ b


# ============================================================================
# act

# -----------------------------------------------
# Tangent geometry (pushforward + kinematic prolongation). The point-geometry
# act paths (Array / Quantity / CDict / product charts) are inherited from
# AbstractLinearTransform.


def _rotate_pushforward_cdict(
    op: "Rotate",
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
) -> CDict:
    """Frozen-tau Jacobian pushforward of tangent data under a rotation.

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
    cart = chart.cartesian
    R = op._matrix(cart, tau)

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


@plum.dispatch
def pushforward(
    op: Rotate,
    tau: Any,
    v: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    *,
    at: CDict | None = None,
    usys: OptUSys = None,
) -> CDict:
    r"""Frozen-$\tau$ pushforward of tangent data under a rotation: $R(\tau) v$.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.transforms as cxfm

    >>> op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> v = {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")}
    >>> out = cxfm.pushforward(op, None, v, cxc.cart3d, cxr.coord_vel)
    >>> out["y"].round(3)
    Q(1., 'm / s')

    """
    return _rotate_pushforward_cdict(op, tau, v, chart, rep, at=at, usys=usys)


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
    at_vel: CDict | None = None,
    usys: OptUSys = None,
    **kw: Any,
) -> CDict:
    r"""Apply a rotation to tangent data (kinematic prolongation).

    - Displacement data and any data under a time-independent rotation
      transform by the frozen-$\tau$ pushforward $v \mapsto R(\tau) v$.
    - Under a time-dependent rotation $R(\tau)$, velocities gain the
      $\dot R$ term of the prolongation, $v' = R v + \dot R x$, which
      requires the base point ``at``; accelerations gain
      $a' = R a + 2 \dot R v + \ddot R x$, requiring ``at`` and ``at_vel``.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.transforms as cxfm
    >>> from jaxtyping import Array, Real

    A uniformly rotating frame (angular speed 1 rad/s about z):

    >>> def R_func(t) -> Real[Array, "3 3"]:
    ...     th = t.ustrip("s")
    ...     st, ct = jnp.sin(th), jnp.cos(th)
    ...     return jnp.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
    >>> op = cxfm.Rotate.from_(R_func)

    At tau=0 the rotation is the identity but the velocity still gains the
    $\dot R x$ (angular) term:

    >>> at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> v = {"x": u.Q(0.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")}
    >>> out = cxfm.act(op, u.Q(0.0, "s"), v, cxc.cart3d, cxr.tangent_geom,
    ...                cxr.coord_vel, at=at)
    >>> out["y"].round(3)
    Q(1., 'm / s')

    """
    del geom, kw

    m = rep.semantic_kind.order
    # Displacements and time-independent rotations: pure pushforward.
    if m == 0 or not callable(op.R):
        return _rotate_pushforward_cdict(op, tau, x, chart, rep, at=at, usys=usys)

    cart = chart.cartesian
    if m == 1 and chart == cart and tau is not None and at is not None:
        # Closed form in Cartesian components: v' = R v + dR/dtau x.
        # One jvp evaluates R(tau) and dR/dtau together.
        tau_val, tau_unit = _tau_value_unit(tau)
        R_fn = op.R
        R, Rdot = jax.jvp(
            lambda tv: R_fn(_attach_leaf(tau_unit, tv)),  # ty: ignore[call-top-callable]
            (tau_val,),
            (jax.numpy.ones_like(tau_val),),
        )
        R = op._validate_shape_match(op._validate_square(R), cart)
        comps = cart.components
        v_arr, v_unit = pack_uniform_unit(x, keys=comps)
        at_arr, at_unit = pack_uniform_unit(at, keys=comps)
        # None units mean "stay raw" throughout, mirroring the generic
        # engine's _attach_leaf/_strip_leaf policy for unitless data.
        Rv = _attach_leaf(v_unit, jnp.einsum("ij,...j->...i", R, v_arr))
        # dR/dtau is per tau's unit; for a raw (unitless) tau with unitful
        # data, interpret it in the data's own time base T = at_unit/v_unit
        # (the same policy as the generic engine's _common_time_unit), so
        # Rdot@at carries at_unit/T = v_unit and the sum is consistent.
        if tau_unit is not None:
            rdot_unit = at_unit / tau_unit if at_unit is not None else None
        elif at_unit is not None and v_unit is not None:
            rdot_unit = v_unit
        else:
            rdot_unit = at_unit
        Rdot_at = _attach_leaf(rdot_unit, jnp.einsum("ij,...j->...i", Rdot, at_arr))
        out_arr = _strip_leaf(v_unit, Rv + Rdot_at)
        if v_unit is None:
            return {k: out_arr[..., i] for i, k in enumerate(comps)}
        return cast("CDict", cxc.cdict(out_arr, v_unit, comps))

    # General case (acceleration, or non-Cartesian chart): generic prolongation
    # (which also owns the missing-tau / missing-anchor errors).
    return prolong_slot(op, tau, x, chart, m, at=at, at_vel=at_vel, usys=usys)
