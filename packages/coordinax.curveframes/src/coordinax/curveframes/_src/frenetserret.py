r"""Frenet--Serret curve-frame data types.

This module provides the concrete implementations of the Frenet--Serret
curve-frame apparatus:

* `FrenetSerretTransform` — a $\tau$-dependent rigid-body transform that
  decomposes into ``Translate(-\gamma) | Rotate([T; N; B])``.
* `FrenetSerretFrame` — a curve-attached reference frame whose axes are the
  Frenet--Serret triad $(\mathbf{T}, \mathbf{N}, \mathbf{B})$.

The transform is constructed from a curve callable via
{meth}`FrenetSerretTransform.from_curve`, which uses JAX automatic
differentiation to compute the first and second derivatives needed for the
tangent, normal, and binormal vectors.

"""

__all__ = ("FrenetSerretFrame", "FrenetSerretTransform")

from collections.abc import Callable
from jaxtyping import Array
from typing import Any, final

import coordinax.charts as cxc
import coordinax.transforms as cxt
import quaxed.numpy as qnp
import unxt as u
from coordinax.internal.custom_types import CDict

from .base import (
    AbstractParallelTransportFrame,
    AbstractParallelTransportTransform,
    FrameT,
)


def _normalize(v: Any) -> Any:
    r"""Normalize a vector to unit length.

    Works transparently with both plain JAX arrays and ``unxt.Quantity``
    objects.  Uses ``quaxed.numpy`` operations so that Quax dispatch handles
    unit-bearing values.

    Parameters
    ----------
    v : array-like or Quantity
        Input vector (any shape with last axis as the vector dimension).

    Returns
    -------
    array-like or Quantity
        Unit vector $\hat{v} = v / \|v\|$.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from coordinax.curveframes._src.frenetserret import _normalize

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
class FrenetSerretTransform(AbstractParallelTransportTransform):
    r"""Transform defined by a Frenet-Serret frame along a curve.

    The Frenet-Serret frame attaches an orthonormal triad $(\mathbf{T},
    \mathbf{N}, \mathbf{B})$ to each point of a smooth space curve
    $\gamma(\tau)$:

    - $\mathbf{T}$ (tangent): unit tangent vector $\gamma'/\|\gamma'\|$
    - $\mathbf{N}$ (normal): unit principal normal $\mathbf{T}'/\|\mathbf{T}'\|$
    - $\mathbf{B}$ (binormal): $\mathbf{T} \times \mathbf{N}$

    Internally decomposes the transform $\mathbf{p}' = R(\tau)(\mathbf{p} -
    \boldsymbol{\gamma}(\tau))$ into ``Translate(-gamma) | Rotate(R)`` where $R
    = [\mathbf{T};\,\mathbf{N};\,\mathbf{B}]$.

    Parameters
    ----------
    translate : Translate
        Tau-dependent translation (callable delta = ``-gamma``).
    rotate : Rotate
        Tau-dependent rotation (callable R = ``stack([T, N, B])``).
    curve : Callable
        The original constructing curve.
    tau_unit : AbstractUnit
        Unit of the curve parameter.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.curveframes as cxfc

    Define a helix:

    >>> def helix(tau: u.Q) -> u.Q:
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), t]), "m")

    Build the transform from the curve:

    >>> fs = cxfc.FrenetSerretTransform.from_curve(helix)
    >>> fs
    FrenetSerretTransform(...)

    Evaluate the location at tau=0:

    >>> loc = fs.location(u.Q(0.0, "s"))
    >>> loc
    Q([1., 0., 0.], 'm')

    """

    # ---------------------------------------------------------------
    # Convenience accessors (tangent inherited from ABC)

    def normal(self, tau: Any) -> u.Q:
        r"""Unit normal vector $\mathbf{N}(\tau)$ (row 1 of R).

        The principal normal lies in the osculating plane and points towards the
        centre of curvature.  It is obtained by Gram--Schmidt rejection of
        $\boldsymbol{\gamma}''$ onto $\mathbf{T}$, then normalised.

        Parameters
        ----------
        tau : Quantity
            The evolution parameter value.

        Returns
        -------
        Quantity
            Dimensionless unit vector of shape ``(3,)``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.curveframes as cxfc

        A circle in the xy-plane: the normal at $\tau=0$ points in the $-x$
        direction (towards the centre).

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> fs = cxfc.FrenetSerretTransform.from_curve(circle)
        >>> fs.normal(u.Q(0.0, "s"))
        Q([-1., -0.,  0.], '')

        """
        R = self._rotation_matrix(tau)
        return u.Q(R[1], "")

    def binormal(self, tau: Any) -> u.Q:
        r"""Unit binormal vector $\mathbf{B}(\tau)$ (row 2 of R).

        The binormal completes the right-handed triad: $\mathbf{B} = \mathbf{T}
        \times \mathbf{N}$.

        Parameters
        ----------
        tau : Quantity
            The evolution parameter value.

        Returns
        -------
        Quantity
            Dimensionless unit vector of shape ``(3,)``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.curveframes as cxfc

        A circle in the xy-plane: the binormal at any $\tau$ points in the $z$
        direction.

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> fs = cxfc.FrenetSerretTransform.from_curve(circle)
        >>> fs.binormal(u.Q(0.0, "s"))
        Q([0., 0., 1.], '')

        """
        R = self._rotation_matrix(tau)
        return u.Q(R[2], "")

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    def from_curve(
        cls,
        curve: Callable[[Any], Any],
        /,
        tau_unit: u.AbstractUnit | str = "s",
    ) -> "FrenetSerretTransform":
        r"""Construct a Frenet-Serret transform from a curve callable.

        Given a smooth curve $\gamma(\tau)$ in 3D Euclidean space, computes the
        Frenet-Serret frame $(\mathbf{T}, \mathbf{N}, \mathbf{B})$ as
        tau-dependent callables using JAX automatic differentiation.

        Derivatives are computed via {func}`unxt.experimental.jacfwd`, which
        correctly tracks physical units through the differentiation.

        Parameters
        ----------
        curve : Callable[[Any], Any]
            A function ``tau -> Quantity[float, (3,)]`` (or Array with 3
            components) representing a smooth space curve.
        tau_unit : str, optional
            The unit of the curve parameter ``tau``, used by
            {func}`unxt.experimental.jacfwd` to compute unit-correct
            derivatives.  Defaults to ``"s"``.

        Returns
        -------
        FrenetSerretTransform
            Transform with lazy (tau-dependent) translate and rotate fields.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.curveframes as cxfc

        A circle in the xy-plane:

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> fs = cxfc.FrenetSerretTransform.from_curve(circle)

        The tangent at tau=0 points in the y-direction:

        >>> T = fs.tangent(u.Q(0.0, "s"))
        >>> T
        Q([-0.,  1.,  0.], '')

        """
        tau_unit = u.unit(tau_unit)  # ty: ignore[invalid-assignment]
        cart = cxc.cart3d

        # Unit-aware first and second derivatives via unxt
        dcurve = u.experimental.jacfwd(curve, units=(tau_unit,))
        d2curve = u.experimental.jacfwd(dcurve, units=(tau_unit,))

        def rotation_matrix_fn(tau: u.AbstractQuantity) -> Any:
            r"""Compute the full rotation matrix R = [T; N; B] with units.

            This closure captures ``dcurve`` and ``d2curve`` (the unit-aware
            first and second derivatives) and evaluates the Frenet--Serret triad
            at a given $\tau$.

            Steps: 1. Evaluate the tangent $\mathbf{T} = \gamma'/\|\gamma'\|$.
            2. Gram--Schmidt: reject $\gamma''$ onto $\mathbf{T}$,
               then normalise to get $\mathbf{N}$.
            3. Cross product: $\mathbf{B} = \mathbf{T} \times \mathbf{N}$.
            4. Stack rows into a $3 \times 3$ matrix.
            """
            dp = dcurve(tau)
            d2p = d2curve(tau)

            # Tangent: normalised first derivative
            t_vec = _normalize(dp)

            # Normal via Gram-Schmidt: remove component of gamma'' along T,
            # then normalise the remainder.
            proj = qnp.sum(d2p * t_vec) * t_vec
            n_unnorm = d2p - proj
            n_vec = _normalize(n_unnorm)

            # Binormal: right-handed completion
            b_vec = qnp.cross(t_vec, n_vec)

            return qnp.stack([t_vec, n_vec, b_vec])

        def rotation_matrix_array_fn(tau: u.AbstractQuantity) -> Array:
            """Rotation matrix as a plain JAX array (strips Quantity units).

            The ``Rotate`` primitive expects a bare numerical array, not a
            ``Quantity``.  This wrapper calls ``rotation_matrix_fn`` and
            extracts ``.value``.
            """
            return rotation_matrix_fn(tau).value

        def neg_gamma_fn(tau: u.AbstractQuantity) -> CDict:
            r"""Compute $-\boldsymbol{\gamma}(\tau)$ as a ``CDict``.

            The translation step of the forward transform subtracts the curve
            position.  This closure wraps the negated curve value into a
            component dictionary keyed by ``cart3d``.
            """
            return cxc.cdict(-curve(tau), cart)  # ty: ignore[invalid-return-type]

        translate = cxt.Translate(neg_gamma_fn, chart=cart)
        rotate = cxt.Rotate(rotation_matrix_array_fn)

        return cls(  # ty: ignore[missing-argument]
            translate=translate,
            rotate=rotate,
            curve=curve,
            tau_unit=tau_unit,
            _is_forward=True,
        )

    # ---------------------------------------------------------------
    # Inverse helpers

    def _build_inverse(self) -> "FrenetSerretTransform":
        """Build the inverse FrenetSerretTransform.

        Constructs a new instance with transposed rotation and rotated
        translation, keeping ``_is_forward=False`` so that a subsequent
        ``.inverse`` call will trigger the clean-rebuild path.
        """
        return FrenetSerretTransform(  # ty: ignore[missing-argument]
            translate=self._make_inverse_translate(),
            rotate=self._make_inverse_rotate(),
            curve=self.curve,
            tau_unit=self.tau_unit,
            _is_forward=False,
        )

    def _rebuild_forward(self) -> "FrenetSerretTransform":
        """Reconstruct the forward transform from the stored curve.

        Called when ``.inverse`` is invoked on an already-inverse instance
        (double-inverse).  Delegates to ``from_curve`` to rebuild from scratch,
        avoiding closure accumulation.
        """
        return FrenetSerretTransform.from_curve(self.curve, tau_unit=self.tau_unit)


#####################################################################
# Frame


@final
class FrenetSerretFrame(AbstractParallelTransportFrame[FrameT]):
    """Frenet-Serret curve-attached reference frame.

    A reference frame defined relative to a base frame by a
    `FrenetSerretTransform`.  At each parameter value ``tau``, the frame is
    centred at the curve position with axes ``(T, N, B)``.

    The evolution parameter ``tau`` is **not** stored on the frame; it is
    supplied at evaluation time via ``act(op, tau, x)``.

    Parameters
    ----------
    base_frame : AbstractReferenceFrame
        The ambient reference frame.
    xop : FrenetSerretTransform
        The tau-dependent rigid-body transform from ``base_frame`` to this
        frame.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.frames as cxf
    >>> import coordinax.transforms as cxfm
    >>> import coordinax.curveframes as cxfc

    >>> def circle(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
    ...                           jnp.zeros_like(t)]), "km")

    Build a frame relative to Alice:

    >>> fs_frame = cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), circle)
    >>> fs_frame.base_frame
    Alice()

    >>> isinstance(fs_frame.xop, cxfc.FrenetSerretTransform)
    True

    Get the frame transition operator and apply at tau=0:

    >>> op = cxf.frame_transition(cxf.Alice(), fs_frame)
    >>> p = u.Q(jnp.array([1.0, 0.0, 0.0]), "km")
    >>> cxfm.act(op, u.Q(0.0, "s"), p)
    Q([0., 0., 0.], 'km')

    """

    base_frame: FrameT
    xop: FrenetSerretTransform
    xop_inv: FrenetSerretTransform

    @classmethod
    def from_curve(
        cls,
        base_frame: FrameT,
        curve: Callable[[Any], Any],
        /,
        tau_unit: u.AbstractUnit | str = "s",
    ) -> "FrenetSerretFrame[FrameT]":
        """Construct a FrenetSerretFrame from a base frame and curve.

        Parameters
        ----------
        base_frame : AbstractReferenceFrame
            The ambient reference frame.
        curve : Callable
            A function ``tau -> Quantity[float, (3,)]`` representing
            a smooth space curve.
        tau_unit : str, optional
            Unit of the curve parameter for differentiation.

        Returns
        -------
        FrenetSerretFrame
            A frame attached to the curve, relative to ``base_frame``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.frames as cxf
        >>> import coordinax.curveframes as cxfc

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "km")

        >>> frame = cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), circle)
        >>> frame.base_frame
        Alice()

        """
        xop = FrenetSerretTransform.from_curve(curve, tau_unit=tau_unit)
        return cls(base_frame=base_frame, xop=xop, xop_inv=xop.inverse)


# ---------------------------------------------------------------
# from_ dispatch


@FrenetSerretTransform.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[FrenetSerretTransform], curve: Callable[[Any], Any], /
) -> FrenetSerretTransform:
    """Construct a FrenetSerretTransform from a curve callable.

    This is the Plum-dispatch convenience constructor. It delegates to
    ``from_curve(curve)`` and therefore uses the default ``tau_unit='s'``.
    """
    return cls.from_curve(curve)
