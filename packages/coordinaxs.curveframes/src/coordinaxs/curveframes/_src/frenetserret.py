r"""Frenet--Serret curve-frame data types.

This module provides the concrete implementations of the Frenet--Serret
curve-frame apparatus:

* `FrenetSerretBuilder` — an `equinox.Module` mapping $\tau$ to the rigid-body
  transform ``Translate(-\gamma) | Rotate([T; N; B])``.
* `FrenetSerretFrame` — a curve-attached reference frame whose axes are the
  Frenet--Serret triad $(\mathbf{T}, \mathbf{N}, \mathbf{B})$.

The frame is constructed from a curve callable via
{meth}`FrenetSerretFrame.from_curve`, which uses JAX automatic differentiation
to compute the first and second derivatives needed for the tangent, normal, and
binormal vectors.

"""

__all__ = ("FrenetSerretBuilder", "FrenetSerretFrame")

from collections.abc import Callable
from jaxtyping import Array
from typing import Any, cast, final

import equinox as eqx

import coordinax.transforms as cxfm
import quaxed.numpy as qnp
import unxt as u
from unxt.quantity import AllowValue

from .base import (
    AbstractCurveFrameBuilder,
    AbstractParallelTransportFrame,
    FrameT,
    unit_or_none,
    unit_tangent,
)

_MSG_ZERO_CURVATURE = (
    "the Frenet--Serret frame cannot be computed where the curvature "
    "vanishes: the normal is the Gram--Schmidt rejection of gamma'' from the "
    "tangent, normalised, and that rejection is 0/0 here. That is every "
    "straight segment and every inflection, not an edge case. Whether the "
    "refusal is *forced* depends on the order at which the rejection "
    "vanishes: at a generic, odd-order inflection the normal flips across the "
    "point and there is no value to return, but at an even-order one the "
    "two-sided limit exists and this refusal is merely conservative -- "
    "recovering it would need the vanishing order, a discrete quantity that "
    "is not JIT-traceable and is itself discontinuous in families. Use "
    "`BishopBuilder`, whose rotation-minimising frame stays defined wherever "
    "the curve is regular; or, on a planar curve, `SignedPlanarBuilder`, "
    "whose normal is continuous through an inflection."
)


_MSG_ZERO_TORSION = (
    "the torsion is undefined where the curvature vanishes: it divides by "
    "|gamma' x gamma''|^2, which is (kappa |gamma'|^3)^2, so a straight "
    "segment or an inflection makes it 0/0. Note that `curvature` IS defined "
    "at those points and reads zero -- only the torsion degenerates. If you "
    "need a frame there rather than an invariant, use `BishopBuilder`, or "
    "`SignedPlanarBuilder` on a planar curve."
)


def _normalize(v: Any) -> Any:
    r"""Normalize a vector to unit length.

    Works transparently with both plain JAX arrays and ``unxt.Quantity``
    objects.  Uses ``quaxed.numpy`` operations so that Quax dispatch handles
    unit-bearing values.

    Parameters
    ----------
    v : array-like or Quantity
        A vector, shape ``(..., 3)``.  The norm is per row -- over the last
        axis only -- so a stacked input normalises each vector separately
        rather than dividing the whole array by one global norm (#953).

    Returns
    -------
    array-like or Quantity
        Unit vector $\hat{v} = v / \|v\|$.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from coordinaxs.curveframes._src.frenetserret import _normalize

    Plain array:

    >>> _normalize(jnp.array([3.0, 4.0, 0.0]))
    Array([0.6, 0.8, 0. ], dtype=float64)

    With units (returns dimensionless after normalisation):

    >>> _normalize(u.Q([0.0, 0.0, 5.0], "m/s"))
    Q([0., 0., 1.], '')

    Row-wise on a stack:

    >>> _normalize(jnp.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]]))
    Array([[0.6, 0.8, 0. ],
           [0. , 0. , 1. ]], dtype=float64)

    """
    norm = qnp.sqrt(qnp.sum(v**2, axis=-1, keepdims=True))
    return v / norm


@final
class FrenetSerretBuilder(AbstractCurveFrameBuilder):
    r"""Frenet--Serret frame family along a curve.

    The Frenet--Serret frame attaches an orthonormal triad $(\mathbf{T},
    \mathbf{N}, \mathbf{B})$ to each point of a smooth space curve
    $\gamma(\tau)$:

    - $\mathbf{T}$ (tangent): unit tangent vector $\gamma'/\|\gamma'\|$
    - $\mathbf{N}$ (normal): unit principal normal $\mathbf{T}'/\|\mathbf{T}'\|$
    - $\mathbf{B}$ (binormal): $\mathbf{T} \times \mathbf{N}$

    Calling the builder at $\tau$ returns the rigid-body transform
    $\mathbf{p}' = R(\tau)(\mathbf{p} - \boldsymbol{\gamma}(\tau))$ decomposed
    as ``Translate(-gamma) | Rotate(R)`` with $R =
    [\mathbf{T};\,\mathbf{N};\,\mathbf{B}]$.

    Parameters
    ----------
    curve : Callable
        A function ``tau -> Quantity[float, (3,)]`` representing a smooth space
        curve.  Make it an `equinox.Module` for differentiable curve parameters;
        a bare function's captures are trace-time constants.
    tau_unit : AbstractUnit or str, optional
        Unit of the curve parameter, used by {func}`unxt.experimental.jacfwd` to
        compute unit-correct derivatives.  `None` (the default) reads it off
        the parameter the builder is called with.  There is no neutral unit to
        default to -- a curve parameter may be a time, an arc length, or an
        affine parameter -- so rather than pick one, take the one the caller
        already stated by passing a `Quantity`.  Declare it for a curve that
        reads its argument's ``.value`` rather than converting, or for a raw
        (unitless) parameter or ``station``, neither of which carries a unit
        to read.
    station : optional
        A fixed station along the curve; see `AbstractCurveFrameBuilder`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinaxs.curveframes as cxfc

    Define a helix:

    >>> def helix(tau: u.Q) -> u.Q:
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), t]), "m")

    >>> fs = cxfc.FrenetSerretBuilder(helix, "s")
    >>> fs.location(u.Q(0.0, "s"))
    Q([1., 0., 0.], 'm')

    """

    curve: Callable[[Any], Any]
    """The constructing curve."""

    tau_unit: u.AbstractUnit | None = eqx.field(
        default=None, static=True, converter=unit_or_none
    )
    """The unit of the curve parameter tau."""

    station: Any = None
    """Optional fixed station along the curve (a leaf); `None` means "use tau"."""

    def rotation_matrix(self, tau: Any, /) -> Array:
        r"""Compute the full rotation matrix $R = [T; N; B]$.

        Steps:

        1. Evaluate the tangent $\mathbf{T} = \gamma'/\|\gamma'\|$.
        2. Gram--Schmidt: reject $\gamma''$ onto $\mathbf{T}$, then normalise to
           get $\mathbf{N}$.
        3. Cross product: $\mathbf{B} = \mathbf{T} \times \mathbf{N}$.
        4. Stack rows into a $3 \times 3$ matrix.

        Notes
        -----
        Where the curvature vanishes this raises rather than returning NaN.
        Whether that refusal is *forced* turns on the order at which the
        Gram--Schmidt rejection of $\gamma''$ vanishes. At a generic,
        odd-order inflection the normal flips sign across the point, so there
        is genuinely nothing to return. At an even-order one it does not: the
        two-sided limit exists and the refusal is conservative.

        Recovering the even-order value would require determining that order,
        which is a discrete quantity -- not JIT-traceable when unknown, and
        discontinuous in families, since $t^4$ deformed to
        $t^3 + \epsilon t^4$ changes the answer abruptly. A generic
        inflection is odd-order, so refusing is right in the common case.
        See GalacticDynamics/coordinax#887.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> fs = cxfc.FrenetSerretBuilder(circle, "s")
        >>> fs.rotation_matrix(u.Q(0.0, "s")).round(3)
        Array([[-0.,  1.,  0.],
               [-1., -0.,  0.],
               [ 0.,  0.,  1.]], dtype=float64)

        """
        # For a two-argument curve `tau` is the time: the apparatus is that
        # of the time slice, at the pinned station. See `_resolve`.
        b, p = self._resolve(tau)

        # Unit-aware first and second derivatives via unxt. `g` is built
        # `_param` resolves the parameter and its unit together, and raises
        # for a bare one before anything reaches `.astype` -- which a unitless
        # parameter does not have, so the other order reported the accident
        # rather than the cause.
        g, tau_unit = b._param(p)
        dcurve = u.experimental.jacfwd(b.curve, units=(tau_unit,))
        d2curve = u.experimental.jacfwd(dcurve, units=(tau_unit,))

        dp = dcurve(g)
        d2p = d2curve(g)

        # Tangent: normalised first derivative
        t_vec = unit_tangent(dp)

        # Normal via Gram-Schmidt: remove component of gamma'' along T,
        # then normalise the remainder.
        # `axis=-1, keepdims=True`: the projection is a per-vector dot product.
        # Summed over every axis it is one global scalar, which is the same
        # number only for a lone 3-vector and silently wrong for a stack (#953).
        proj = qnp.sum(d2p * t_vec, axis=-1, keepdims=True) * t_vec
        n_unnorm = d2p - proj
        # Relative to |gamma''|, as `bishop._orthonormalize` guards its own
        # rejection: a vanishing rejection only means something against the size
        # of what was rejected. As a ratio it is dimensionless, which `error_if`
        # needs. `~(x > tol)`, not `x <= tol`: NaN is False for both, and a
        # straight segment gives `0/0 = nan`.
        ratio = qnp.sqrt(qnp.sum(n_unnorm**2, axis=-1)) / qnp.sqrt(
            qnp.sum(d2p**2, axis=-1)
        )
        n_unnorm = eqx.error_if(
            n_unnorm,
            ~(cast("Array", u.ustrip(AllowValue, "", ratio)) > 1e-12),
            _MSG_ZERO_CURVATURE,
        )
        n_vec = _normalize(n_unnorm)

        # Binormal: right-handed completion
        b_vec = qnp.cross(t_vec, n_vec)

        # ``Rotate`` expects a bare numerical array, not a ``Quantity``.
        return qnp.stack([t_vec, n_vec, b_vec]).value  # ty: ignore[unresolved-attribute]

    def tangent(self, tau: Any, /) -> u.Q:
        r"""Return the unit tangent vector $\mathbf{T}(\tau)$ (row 0 of R).

        Overrides the base implementation, which would take row 0 of the full
        rotation matrix — and so pay for $\boldsymbol{\gamma}''$, which only
        $\mathbf{N}$ and $\mathbf{B}$ need. The value is identical.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> cxfc.FrenetSerretBuilder(circle, "s").tangent(u.Q(0.0, "s"))
        Q([-0.,  1.,  0.], '')

        """
        b, p = self._resolve(tau)
        g, tau_unit = b._param(p)
        dcurve = u.experimental.jacfwd(b.curve, units=(tau_unit,))
        return u.Q(unit_tangent(dcurve(g)).value, "")

    def normal(self, tau: Any, /) -> u.Q:
        r"""Return the unit normal vector $\mathbf{N}(\tau)$ (row 1 of R).

        The principal normal lies in the osculating plane and points towards the
        centre of curvature.  It is obtained by Gram--Schmidt rejection of
        $\boldsymbol{\gamma}''$ onto $\mathbf{T}$, then normalised.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        A circle in the xy-plane: the normal at $\tau=0$ points in the $-x$
        direction (towards the centre).

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> cxfc.FrenetSerretBuilder(circle, "s").normal(u.Q(0.0, "s"))
        Q([-1., -0.,  0.], '')

        """
        return u.Q(self.rotation_matrix(tau)[1], "")

    def binormal(self, tau: Any, /) -> u.Q:
        r"""Return the unit binormal vector $\mathbf{B}(\tau)$ (row 2 of R).

        The binormal completes the right-handed triad: $\mathbf{B} = \mathbf{T}
        \times \mathbf{N}$.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        A circle in the xy-plane: the binormal at any $\tau$ points in the $z$
        direction.

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> cxfc.FrenetSerretBuilder(circle, "s").binormal(u.Q(0.0, "s"))
        Q([0., 0., 1.], '')

        """
        return u.Q(self.rotation_matrix(tau)[2], "")

    def curvature(self, tau: Any, /) -> u.Q:
        r"""Return the curvature $\kappa(\tau) \ge 0$.

        $$ \kappa = \frac{\|\boldsymbol{\gamma}' \times
           \boldsymbol{\gamma}''\|}{\|\boldsymbol{\gamma}'\|^3} $$

        **Defined where the frame is not.** `rotation_matrix` refuses at an
        inflection and on a straight segment, because the *normal* has no
        direction there. The curvature has no such problem: it is a
        non-negative scalar that simply reads zero. Guarding it would refuse a
        correct answer, so it is unguarded.

        Returns a `Quantity` of dimension 1/length. Costs two `jacfwd` passes,
        the same as `rotation_matrix`; only `torsion` pays for a third.

        On a planar curve this is $|\kappa_s|$, the magnitude of
        `SignedPlanarBuilder.signed_curvature`.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        A straight line, where the *frame* is undefined everywhere:

        >>> def line(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([t, jnp.zeros_like(t),
        ...                           jnp.zeros_like(t)]), "km")

        >>> float(cxfc.FrenetSerretBuilder(line, "s").curvature(u.Q(2.0, "s"))
        ...       .ustrip("1/km"))
        0.0

        """
        b, p = self._resolve(tau)
        g, tau_unit = b._param(p)
        dcurve = u.experimental.jacfwd(b.curve, units=(tau_unit,))
        d2curve = u.experimental.jacfwd(dcurve, units=(tau_unit,))
        dp = dcurve(g)
        d2p = d2curve(g)
        return qnp.sqrt(qnp.sum(qnp.cross(dp, d2p) ** 2)) / (
            qnp.sqrt(qnp.sum(dp**2)) ** 3
        )

    def torsion(self, tau: Any, /) -> u.Q:
        r"""Return the torsion: how fast the curve leaves its osculating plane.

        $$ \tau_g = \frac{(\boldsymbol{\gamma}' \times
           \boldsymbol{\gamma}'') \cdot \boldsymbol{\gamma}'''}
           {\|\boldsymbol{\gamma}' \times \boldsymbol{\gamma}''\|^2} $$

        Unlike `curvature`, this **is** undefined where the curvature
        vanishes: the denominator is $(\kappa \|\gamma'\|^3)^2$, so a straight
        segment or an inflection gives $0/0$. It is refused there rather than
        returned as NaN, in the same way `rotation_matrix` refuses.

        Zero on any planar curve, wherever it is defined.

        Returns a `Quantity` of dimension 1/length. Costs a third `jacfwd`
        pass, which is why it is a separate accessor: `rotation_matrix` never
        pays for it.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        A helix ``(a cos t, a sin t, b t)`` has torsion ``b / (a^2 + b^2)``:

        >>> def helix(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")

        >>> float(cxfc.FrenetSerretBuilder(helix, "s").torsion(u.Q(0.7, "s"))
        ...       .ustrip("1/km").round(6))
        0.275229

        """
        b, p = self._resolve(tau)
        g, tau_unit = b._param(p)
        dcurve = u.experimental.jacfwd(b.curve, units=(tau_unit,))
        d2curve = u.experimental.jacfwd(dcurve, units=(tau_unit,))
        d3curve = u.experimental.jacfwd(d2curve, units=(tau_unit,))
        dp = dcurve(g)
        d2p = d2curve(g)
        d3p = d3curve(g)

        cross = qnp.cross(dp, d2p)
        # Guard on the *relative* magnitude, as `rotation_matrix` does: this
        # ratio is the sine of the angle between gamma' and gamma'', so it is
        # dimensionless (which `error_if` needs) and vanishes exactly when the
        # two are parallel -- which is exactly when kappa = 0. `~(x > tol)`,
        # not `x <= tol`: a straight segment gives `0/0 = nan`, and NaN is
        # False for both. The *checked* value is what the result is built
        # from, since an `error_if` whose result is dropped is dead code.
        ratio = qnp.sqrt(qnp.sum(cross**2)) / (
            qnp.sqrt(qnp.sum(dp**2)) * qnp.sqrt(qnp.sum(d2p**2))
        )
        cross = eqx.error_if(
            cross,
            ~(cast("Array", u.ustrip(AllowValue, "", ratio)) > 1e-12),
            _MSG_ZERO_TORSION,
        )
        return qnp.sum(cross * d3p) / qnp.sum(cross**2)


#####################################################################
# Frame


@final
class FrenetSerretFrame(AbstractParallelTransportFrame[FrameT]):
    """Frenet-Serret curve-attached reference frame.

    A reference frame defined relative to a base frame by a
    `coordinax.transforms.TimeDep` wrapping a `FrenetSerretBuilder`.  At each
    parameter value ``tau``, the frame is centred at the curve position with
    axes ``(T, N, B)``.

    The evolution parameter ``tau`` is **not** stored on the frame; it is
    supplied at evaluation time via ``act(op, tau, x)``.

    Parameters
    ----------
    base_frame : AbstractReferenceFrame
        The ambient reference frame.
    xop : TimeDep
        The tau-dependent rigid-body transform from ``base_frame`` to this
        frame.
    xop_inv : TimeDep
        Its inverse.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.frames as cxf
    >>> import coordinax.transforms as cxfm
    >>> import coordinaxs.curveframes as cxfc

    >>> def circle(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
    ...                           jnp.zeros_like(t)]), "km")

    Build a frame relative to Alice:

    >>> fs_frame = cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), circle, "s")
    >>> fs_frame.base_frame
    Alice()

    >>> isinstance(fs_frame.xop.builder, cxfc.FrenetSerretBuilder)
    True

    Get the frame transition operator and apply at tau=0:

    >>> op = cxf.frame_transition(cxf.Alice(), fs_frame)
    >>> p = u.Q(jnp.array([1.0, 0.0, 0.0]), "km")
    >>> cxfm.act(op, u.Q(0.0, "s"), p)
    Q([0., 0., 0.], 'km')

    """

    base_frame: FrameT
    xop: cxfm.TimeDep
    xop_inv: cxfm.TimeDep

    @classmethod
    def from_curve(
        cls,
        base_frame: FrameT,
        curve: Callable[[Any], Any],
        /,
        tau_unit: u.AbstractUnit | str | None = None,
        *,
        station: Any = None,
    ) -> "FrenetSerretFrame[FrameT]":
        """Construct a FrenetSerretFrame from a base frame and curve.

        Parameters
        ----------
        base_frame : AbstractReferenceFrame
            The ambient reference frame.
        curve : Callable
            A function ``tau -> Quantity[float, (3,)]`` representing
            a smooth space curve.
        tau_unit : AbstractUnit or str, optional
            Unit of the curve parameter for differentiation.  `None` (the
            default) reads it off the parameter the frame is evaluated at.
            There is no neutral unit to default to -- a curve parameter may be
            a time, an arc length, or an affine parameter -- so rather than
            pick one, take the one the caller already stated by passing a
            `Quantity`.
        station : optional
            A fixed station along the curve; when given the frame is a fixed
            frame *field* along the curve rather than a moving frame.

        Returns
        -------
        FrenetSerretFrame
            A frame attached to the curve, relative to ``base_frame``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.frames as cxf
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "km")

        >>> frame = cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), circle, "s")
        >>> frame.base_frame
        Alice()

        """
        builder = FrenetSerretBuilder(curve, tau_unit, station)
        xop = cxfm.TimeDep(builder)
        return cls(base_frame=base_frame, xop=xop, xop_inv=xop.inverse)
