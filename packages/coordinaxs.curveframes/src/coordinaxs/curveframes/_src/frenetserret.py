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
from typing import Any, final

import equinox as eqx

import coordinax.transforms as cxfm
import quaxed.numpy as qnp
import unxt as u

from .base import AbstractCurveFrameBuilder, AbstractParallelTransportFrame, FrameT


def _normalize(v: Any) -> Any:
    r"""Normalize a vector to unit length.

    Works transparently with both plain JAX arrays and ``unxt.Quantity``
    objects.  Uses ``quaxed.numpy`` operations so that Quax dispatch handles
    unit-bearing values.

    Parameters
    ----------
    v : array-like or Quantity
        A single vector (1-D).  The norm is taken over *all* axes, so batched
        input is NOT supported; every caller here passes one 3-vector.

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

    """
    norm = qnp.sqrt(qnp.sum(v**2))
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
    tau_unit : AbstractUnit or str
        Unit of the curve parameter, used by {func}`unxt.experimental.jacfwd` to
        compute unit-correct derivatives.  Required: there is no neutral
        default, since a curve parameter may be a time, an arc length, or an
        affine parameter, and the wrong unit is silently rescaled rather than
        rejected when it is dimensionally compatible.
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

    tau_unit: u.AbstractUnit = eqx.field(  # ty: ignore[invalid-assignment]
        static=True, converter=u.unit
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

        # Unit-aware first and second derivatives via unxt
        dcurve = u.experimental.jacfwd(b.curve, units=(b.tau_unit,))
        d2curve = u.experimental.jacfwd(dcurve, units=(b.tau_unit,))

        g = b._param(p).astype(float)
        dp = dcurve(g)
        d2p = d2curve(g)

        # Tangent: normalised first derivative
        t_vec = _normalize(dp)

        # Normal via Gram-Schmidt: remove component of gamma'' along T,
        # then normalise the remainder.
        proj = qnp.sum(d2p * t_vec) * t_vec
        n_unnorm = d2p - proj
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
        dcurve = u.experimental.jacfwd(b.curve, units=(b.tau_unit,))
        return u.Q(_normalize(dcurve(b._param(p).astype(float))).value, "")

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
        tau_unit: u.AbstractUnit | str,
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
        tau_unit : AbstractUnit or str
            Unit of the curve parameter for differentiation.  Required: there
            is no neutral default, since a curve parameter may be a time, an
            arc length, or an affine parameter, and the wrong unit is silently
            rescaled rather than rejected when it is dimensionally compatible.
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
