"""Galilean coordinate transformations."""

__all__ = ("Rotate",)


from dataclasses import replace

from jaxtyping import Array, Shaped
from typing import Any, Final, final

import equinox as eqx
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
from .linear import AbstractLinearTransform, as_dimensionless_matrix
from .utils import is_traced
from coordinax.internal import pack_uniform_unit
from coordinax.transforms._src import groups

_ATOL: Final = 1e-6
"""Absolute tolerance on ``R^T R = I``. See `_not_a_rotation`."""

_MSG_NOT_A_ROTATION: Final = (
    "Rotate requires a rotation matrix: R^T R = I with det R = +1, i.e. SO(n). "
    "Orthogonality is what `inverse` relies on -- it transposes. For a "
    "hyperplane reflection -- orthogonal, det = -1 *and* an involution -- use "
    "`Reflect`; for any other invertible linear map use `Linear`. Note a "
    "rotoreflection is orthogonal with det = -1 and is not an involution, so "
    "it belongs in `Linear`, not `Reflect`."
)


def _not_a_rotation(R: Any, /) -> Any:
    """Whether ``R`` fails ``R^T R = I`` with ``det R = +1``.

    A non-square ``R`` answers `False`: it has no transpose product to compare,
    and `_validate_square` is the one that names a bad shape. The shape is
    static under tracing, so this branch traces.

    ``atol`` is explicit rather than `jnp.allclose`'s ``1e-8``, which is below
    the round-off of an honest rotation matrix. The off-diagonal entries are
    compared against zero, where ``rtol`` contributes nothing, so ``1e-8`` is
    the whole budget -- and a parallel-transported Bishop triad
    (`coordinaxs.curveframes`) drifts to ~``5e-8`` off orthogonal, which is
    why that package's own doctests assert orthogonality at ``1e-6``. The same
    number here. It is many orders away from catching less: the matrix in #938
    has ``R^T R`` entries in the tens.
    """
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False
    gram = jnp.matmul(jnp.swapaxes(R, -2, -1), R)
    orthogonal = jnp.allclose(gram, jnp.eye(R.shape[0], dtype=gram.dtype), atol=_ATOL)
    # `det R = -1` is orthogonal but orientation-reversing: a reflection or a
    # rotoreflection, not a rotation. `Reflect` and `Linear` are those homes.
    proper = jnp.allclose(jnp.linalg.det(R), 1.0, atol=_ATOL)
    return ~(orthogonal & proper)


def _as_rotation_matrix(R: Any, /) -> Array:
    """Normalise ``R`` to a bare array, requiring it to be dimensionless.

    A rotation matrix preserves lengths, so its entries are ratios: a
    dimensionless quantity is stripped and anything else refused. Shares
    `as_dimensionless_matrix` with `Reflect` and `Shear`.
    """
    return as_dimensionless_matrix(
        R,
        "Rotate `R` is a rotation matrix, whose entries are ratios and "
        "so dimensionless.",
    )


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
    R : Array[float, (N, N)]
        The rotation matrix.

    Raises
    ------
    equinox.EquinoxTracetimeError
        If ``R`` is not square. A shape is static, so this is decided while
        tracing and raises there -- not when the traced graph runs.
    equinox.EquinoxRuntimeError
        If ``R`` is not a rotation -- ``R^T R = I`` *and* ``det R = +1``, i.e.
        SO(N). Orthogonality alone is not enough: an improper orthogonal
        matrix reverses orientation, and `Reflect` or `Linear` is its home.
        This one depends on the values, so it is deferred onto the stored
        ``R`` to survive `jax.jit`: eagerly it raises from the constructor,
        under `jit` when the traced graph runs.

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
    >>> nq
    {'x': Q(0, 'm'), 'y': Q(1, 'm'), 'z': Q(0, 'm')}

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

    ``R`` is always a constant matrix. A time-dependent rotation is a
    `~coordinax.transforms.TimeDep` family of `Rotate` operators — e.g.
    built by `~coordinax.transforms.builders.RotationAboutAxis`:

    >>> zhat = jnp.array([0.0, 0.0, 1.0])
    >>> b = cxfm.builders.RotationAboutAxis(u.Q(45, "deg/s"), axis=zhat)
    >>> R_op = cxfm.TimeDep(b)

    >>> t = u.Q(4, "s")  # 180 degrees rotation
    >>> R_op(t, q).round(3)
    Q([[-1.,  0.,  0.],
       [-0., -1.,  0.]], 'm')

    """

    R: Shaped[Array, " N N"] = eqx.field(converter=_as_rotation_matrix)
    """The rotation matrix."""

    @classmethod
    def groups(cls) -> frozenset[type]:
        """Return the groups to which this map belongs.

        `~coordinax.transforms.groups.SpecialOrthogonalGroup` unconditionally:
        the constructor admits only ``R^T R = I`` with ``det R = +1``, so there
        is no determinant to read and nothing to decide per instance.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import coordinax.transforms as cxfm

        >>> Rz = jnp.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        >>> sorted(g.__name__ for g in cxfm.Rotate(Rz).groups())
        ['DiffeomorphismGroup', 'SpecialOrthogonalGroup']

        """
        return frozenset((groups.SpecialOrthogonalGroup, groups.DiffeomorphismGroup))

    def __init__(self, R: Any) -> None:
        # Through the field converter, not `quaxed.numpy.asarray`: the
        # converter is what strips (or refuses) units, and `jnp.asarray` hands
        # a `~unxt.Quantity` straight back. Equinox re-applies the converter
        # after this returns, so the only thing that changed is *when* -- and
        # the orthogonality check below needs a bare array to test.
        R = _as_rotation_matrix(R)
        # Deferred so it survives jit (a plain `bool` on a traced value raises
        # `TracerBoolConversionError`), and threaded onto the stored array so
        # it is not dead-code-eliminated: `inverse` transposes instead of
        # inverting, which is the inverse only for an orthogonal `R`.
        # Shape first: `_not_a_rotation` declines on a non-square matrix (it
        # has no `R^T R` to compare), so without this a non-square `R` would be
        # stored and `.inverse` would hand back a meaningless transpose.
        R = self._validate_square(R)
        object.__setattr__(
            self, "R", eqx.error_if(R, _not_a_rotation(R), _MSG_NOT_A_ROTATION)
        )

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
        return replace(self, R=jnp.swapaxes(self.R, -2, -1))

    # -----------------------------------------------------

    @property
    def _raw_matrix(self) -> Any:
        return self.R

    # -----------------------------------------------------
    # Arithmetic operations

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
        Rotate(f64[3,3](jax))

        >>> jnp.allclose(op3.R, op2.R @ op1.R)
        Array(True, dtype=bool)

        """
        if not isinstance(other, Rotate):
            return NotImplemented
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

    The identity check inspects values, so it is skipped when ``approx=False``,
    and when the matrix is traced -- under `jax.jit` the values are not known,
    which is exactly when the answer is "do not simplify" rather than an error.

    """
    if (
        approx
        and not is_traced(op.R)
        and jnp.allclose(op.R, jnp.eye(op.R.shape[-1], dtype=op.R.dtype), **kw)
    ):
        return identity
    return op


@plum.dispatch
def _merge(a: Rotate, b: Rotate, /) -> AbstractTransform | None:
    """Merge two adjacent rotations (``a`` applied first) into one, as ``a @ b``."""
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
