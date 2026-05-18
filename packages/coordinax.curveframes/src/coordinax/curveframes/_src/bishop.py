r"""Bishop (rotation-minimising) curve-frame data types.

This module provides the concrete implementations of the Bishop
(parallel-transport, rotation-minimising) curve-frame apparatus:

* `BishopTransform` — a $\tau$-dependent rigid-body transform that decomposes
  into ``Translate(-\gamma) | Rotate([T; U1; U2])``.
* `BishopFrame` — a curve-attached reference frame whose axes are $(\mathbf{T},
  \mathbf{U}_1, \mathbf{U}_2)$ obtained by parallel transport along the curve.

Unlike the Frenet--Serret frame, the Bishop frame is **well-defined even when
the curvature vanishes** ($\kappa = 0$), because it does not depend on
$\boldsymbol{\gamma}''$.  The normal-plane vectors are obtained by solving the
parallel-transport ODE:

$$ \frac{d\mathbf{U}_i}{d\tau}
  = -\bigl(\mathbf{U}_i \cdot \mathbf{T}'\bigr)\,\mathbf{T}, \qquad i \in \{1,
  2\},
$$

starting from an initial orthonormal pair at a reference parameter $\tau_0$.
The ODE is integrated numerically using {func}`jax.experimental.ode.odeint`.

Both classes are ``@final`` (no further subclassing).

Key design choices
------------------
* **Lazy evaluation** — all frame vectors are $\tau$-dependent callables.  The
  ODE is solved only when a concrete $\tau$ is requested.
* **Double-inverse efficiency** — same two-step cycle as
  ``FrenetSerretTransform``.
* **Auto initial normal** — when no ``initial_normal`` is supplied, one is
  chosen automatically via Gram--Schmidt against the tangent at $\tau_0$.

See Also
--------
coordinax.curveframes._src.frenetserret : Frenet--Serret frame.
coordinax.curveframes._src.base : Abstract base classes.
"""

__all__ = ("BishopFrame", "BishopTransform")

from collections.abc import Callable
from typing import Any, cast, final

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

import coordinax.charts as cxc
import coordinax.transforms as cxt
import unxt as u

from .base import (
    AbstractParallelTransportFrame,
    AbstractParallelTransportTransform,
    FrameT,
)
from .frenetserret import _normalize


def _auto_initial_normal(T0_val: Any) -> Any:
    r"""Choose an initial normal vector via Gram--Schmidt projection.

    Given the unit tangent $\mathbf{T}_0$ at the reference parameter $\tau_0$,
    this function selects the standard basis vector $\mathbf{e}_k$ that is
    **least aligned** with $\mathbf{T}_0$ (i.e. $k = \arg\min_j\, |\mathbf{T}_0
    \cdot \mathbf{e}_j|$), then projects out the tangent component and
    normalises the result.

    This guarantees a numerically stable initial normal even when $\mathbf{T}_0$
    is closely aligned with one of the coordinate axes.

    Parameters
    ----------
    T0_val : array-like, shape ``(3,)``
        Dimensionless unit tangent vector at $\tau_0$.

    Returns
    -------
    array, shape ``(3,)``
        Dimensionless unit normal $\mathbf{U}_{1,0}$.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from coordinax.curveframes._src.bishop import _auto_initial_normal

    When the tangent is along x, the initial normal avoids x:

    >>> T0 = jnp.array([1.0, 0.0, 0.0])
    >>> U1 = _auto_initial_normal(T0)
    >>> float(jnp.dot(U1, T0))
    0.0
    >>> float(jnp.linalg.norm(U1))
    1.0

    """
    # Pick the standard basis axis least aligned with T0.
    # This maximises the rejection magnitude and avoids numerical
    # cancellation in the Gram--Schmidt step.
    abs_T0 = jnp.abs(T0_val)
    k = jnp.argmin(abs_T0)
    e_k = jnp.zeros(3).at[k].set(1.0)

    # Gram--Schmidt: subtract the component along T0, then normalise.
    proj = jnp.dot(e_k, T0_val) * T0_val
    u1 = e_k - proj
    return u1 / jnp.linalg.norm(u1)


@final
class BishopTransform(AbstractParallelTransportTransform):
    r"""Transform defined by a Bishop (rotation-minimising) frame along a curve.

    The Bishop frame attaches an orthonormal triad $(\mathbf{T}, \mathbf{U}_1,
    \mathbf{U}_2)$ to each point of a smooth space curve $\gamma(\tau)$ via
    parallel transport:

    - $\mathbf{T}$ (tangent): unit tangent vector $\gamma'/\|\gamma'\|$
    - $\mathbf{U}_1$ (normal 1): parallel-transported first normal
    - $\mathbf{U}_2$ (normal 2): $\mathbf{T} \times \mathbf{U}_1$

    Unlike the Frenet-Serret frame, the Bishop frame is well-defined even when
    the curvature vanishes ($\kappa = 0$).

    Internally decomposes the transform $\mathbf{p}' = R(\tau)(\mathbf{p} -
    \boldsymbol{\gamma}(\tau))$ into ``Translate(-gamma) | Rotate(R)`` where $R
    = [\mathbf{T};\,\mathbf{U}_1;\,\mathbf{U}_2]$.

    Parameters
    ----------
    translate : Translate
        Tau-dependent translation (callable delta = ``-gamma``).
    rotate : Rotate
        Tau-dependent rotation (callable R = ``stack([T, U1, U2])``).
    curve : Callable
        The original constructing curve.
    tau_unit : AbstractUnit
        Unit of the curve parameter.
    tau_0 : AbstractQuantity
        Reference parameter value where the initial frame is defined.
    initial_normal : Any
        Initial U1 vector at tau_0 (dimensionless jax array, or None for auto).

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

    >>> bt = cxfc.BishopTransform.from_curve(helix)
    >>> bt
    BishopTransform(...)

    Evaluate the location at tau=0:

    >>> loc = bt.location(u.Q(0.0, "s"))
    >>> loc
    Q([1., 0., 0.], 'm')

    The Bishop frame works on a straight line (where Frenet-Serret is singular):

    >>> def line(tau: u.Q) -> u.Q:
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([t, jnp.zeros_like(t),
    ...                           jnp.zeros_like(t)]), "m")

    >>> bt_line = cxfc.BishopTransform.from_curve(line)
    >>> U1 = bt_line.normal1(u.Q(5.0, "s"))
    >>> jnp.sqrt(jnp.sum(U1.value**2))
    Array(1., dtype=float64)

    """

    tau_0: u.AbstractQuantity
    """Reference parameter value where the initial frame is defined."""

    initial_normal: Any
    """Initial U1 vector at tau_0 (dimensionless jax array, or None for auto)."""

    # ---------------------------------------------------------------
    # Convenience accessors (tangent inherited from ABC)

    def normal1(self, tau: Any) -> Any:
        r"""First parallel-transported normal $\mathbf{U}_1(\tau)$ (row 1 of R).

        This vector is obtained by solving the parallel-transport ODE from the
        reference parameter $\tau_0$.  It is perpendicular to the tangent and
        rotation-minimising: the angular velocity of the frame about the tangent
        is zero.

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

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> bt = cxfc.BishopTransform.from_curve(circle)
        >>> U1 = bt.normal1(u.Q(0.0, "s"))
        >>> float(jnp.linalg.norm(U1.value))
        1.0

        """
        R = self._rotation_matrix(tau)
        return u.Q(R[1], "")

    def normal2(self, tau: Any) -> Any:
        r"""Second normal $\mathbf{U}_2(\tau) = \mathbf{T} \times \mathbf{U}_1$.

        The second normal completes the right-handed orthonormal triad.  It is
        computed as the cross product of the tangent and the first normal, so it
        is automatically perpendicular to both.

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

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> bt = cxfc.BishopTransform.from_curve(circle)
        >>> U2 = bt.normal2(u.Q(0.0, "s"))
        >>> float(jnp.linalg.norm(U2.value))
        1.0

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
        *,
        tau_0: u.AbstractQuantity | None = None,
        initial_normal: Any | None = None,
    ) -> "BishopTransform":
        r"""Construct a Bishop transform from a curve callable.

        Given a smooth curve $\gamma(\tau)$ in 3D Euclidean space, computes the
        Bishop frame $(\mathbf{T}, \mathbf{U}_1, \mathbf{U}_2)$ as tau-dependent
        callables.  The tangent is obtained via JAX automatic differentiation.
        The normal vectors are computed by solving the parallel-transport ODE
        using ``jax.experimental.ode.odeint``.

        Parameters
        ----------
        curve : Callable[[Any], Any]
            A function ``tau -> Quantity[float, (3,)]``.
        tau_unit : str, optional
            Unit of the curve parameter.  Defaults to ``"s"``.
        tau_0 : Quantity, optional
            Reference parameter for the initial frame.  Defaults to ``Q(0.0,
            tau_unit)``.
        initial_normal : array-like, optional
            Dimensionless 3-vector for $\mathbf{U}_{1,0}$.  When ``None``,
            auto-chosen via Gram-Schmidt.

        Returns
        -------
        BishopTransform

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.curveframes as cxfc

        A circle in the xy-plane:

        >>> def circle(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> bt = cxfc.BishopTransform.from_curve(circle)

        The tangent at tau=0 points in the y-direction:

        >>> T = bt.tangent(u.Q(0.0, "s"))
        >>> jnp.allclose(T.value, jnp.array([0., 1., 0.]), atol=1e-5)
        Array(True, dtype=bool)

        Custom tau_0:

        >>> bt2 = cxfc.BishopTransform.from_curve(circle, tau_0=u.Q(1.0, "s"))
        >>> bt2.tau_0
        Q(1., 's')

        """
        tau_unit = cast(u.AbstractUnit, u.unit(tau_unit))
        if tau_0 is None:
            tau_0 = u.Q(0.0, tau_unit)

        # Unit-aware derivative
        dcurve = u.experimental.jacfwd(curve, units=(tau_unit,))

        def tangent_fn(tau: u.AbstractQuantity) -> u.AbstractQuantity:
            r"""Compute unit tangent $\mathbf{T} = \gamma'/\|\gamma'\|$."""
            return _normalize(dcurve(tau.astype(float)))

        # Compute initial tangent and normal at the reference parameter.
        T0 = tangent_fn(tau_0)  # dimensionless unit vector
        T0_val = T0.value  # dimensionless plain array

        if initial_normal is not None:
            U1_0_val = jnp.asarray(initial_normal, dtype=float)
        else:
            U1_0_val = _auto_initial_normal(T0_val)

        # Store the initial normal for double-inverse reconstruction.
        # ``None`` means "auto-chosen via Gram--Schmidt".
        stored_initial_normal = initial_normal  # None for auto

        # Pre-compute dT/dtau as a callable.  This avoids nesting AD inside
        # the ODE right-hand-side, which would be both slower and harder
        # for JAX to trace.
        dTangent_fn = u.experimental.jacfwd(tangent_fn, units=(tau_unit,))

        def _solve_U1(tau: u.AbstractQuantity) -> Any:
            r"""Compute $\mathbf{U}_1(\tau)$ via ODE integration from $\tau_0$.

            Solves the parallel-transport ODE $d\mathbf{U}_1/d\tau =
            -(\mathbf{U}_1 \cdot \mathbf{T}')\,\mathbf{T}$ using
            ``jax.experimental.ode.odeint``.  When $\tau = \tau_0$, the ODE is
            skipped via ``jax.lax.cond`` (identity path).
            """
            tau_val = tau.ustrip(tau_unit)
            tau_0_val = tau_0.ustrip(tau_unit)

            def ode_rhs(U1_flat: Any, t_scalar: Any) -> Any:
                """Right-hand side of the parallel-transport ODE."""
                t_q = u.Q(t_scalar, tau_unit)
                T_val = tangent_fn(t_q).value
                dT_val = dTangent_fn(t_q).value
                # Project U1 onto dT, negate, then scale by T.
                return -jnp.dot(U1_flat, dT_val) * T_val

            # Use lax.cond to branch: when tau == tau_0, return initial
            # normal directly (avoids zero-length ODE integration).
            needs_ode = jnp.abs(tau_val - tau_0_val) > 0.0  # ty: ignore[unsupported-operator]

            def _solve(_: Any) -> Any:
                t_span = jnp.array([tau_0_val, tau_val])
                result = odeint(ode_rhs, U1_0_val, t_span)
                return result[-1]  # solution at tau

            def _identity(_: Any) -> Any:
                return U1_0_val

            U1_val = jax.lax.cond(needs_ode, _solve, _identity, None)
            # Re-normalise for numerical safety.
            return U1_val / jnp.linalg.norm(U1_val)

        def rotation_matrix_fn(tau: u.AbstractQuantity) -> Any:
            r"""Compute the rotation $R = [T;\,U_1;\,U_2]$.

            Steps:

            1. Evaluate the tangent $\mathbf{T}$ from the first derivative.
            2. Solve the parallel-transport ODE for $\mathbf{U}_1$.
            3. Cross product: $\mathbf{U}_2 = \mathbf{T} \times \mathbf{U}_1$.
            4. Stack rows into a $3 \times 3$ matrix.
            """
            T_val = tangent_fn(tau).value
            U1_val = _solve_U1(tau)
            U2_val = jnp.cross(T_val, U1_val)
            return jnp.stack([T_val, U1_val, U2_val])

        def neg_gamma_fn(tau: u.AbstractQuantity) -> Any:
            r"""Compute $-\boldsymbol{\gamma}(\tau)$ as a ``CDict``.

            The translation step of the forward transform subtracts the curve
            position, packed into a component dictionary.
            """
            return cxc.cdict(-curve(tau), cxc.cart3d)

        translate = cxt.Translate(neg_gamma_fn, chart=cxc.cart3d)
        rotate = cxt.Rotate(rotation_matrix_fn)

        return cls(  # ty: ignore[missing-argument]
            translate=translate,
            rotate=rotate,
            curve=curve,
            tau_unit=tau_unit,
            _is_forward=True,
            tau_0=tau_0,
            initial_normal=stored_initial_normal,
        )

    # ---------------------------------------------------------------
    # Inverse helpers

    def _build_inverse(self) -> "BishopTransform":
        """Build the inverse BishopTransform.

        Constructs a new instance with transposed rotation and rotated
        translation, keeping ``_is_forward=False`` so that a subsequent
        ``.inverse`` call will trigger the clean-rebuild path.  Preserves
        ``tau_0`` and ``initial_normal`` for double-inverse reconstruction.
        """
        return BishopTransform(  # ty: ignore[missing-argument]
            translate=self._make_inverse_translate(),
            rotate=self._make_inverse_rotate(),
            curve=self.curve,
            tau_unit=self.tau_unit,
            _is_forward=False,
            tau_0=self.tau_0,
            initial_normal=self.initial_normal,
        )

    def _rebuild_forward(self) -> "BishopTransform":
        """Reconstruct the forward transform from the stored curve.

        Called when ``.inverse`` is invoked on an already-inverse instance
        (double-inverse).  Delegates to ``from_curve`` with the stored ``tau_0``
        and ``initial_normal``, avoiding closure accumulation.
        """
        return BishopTransform.from_curve(
            self.curve,
            tau_unit=self.tau_unit,
            tau_0=self.tau_0,
            initial_normal=self.initial_normal,
        )


# ============================================================================
# Constructors


@BishopTransform.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[BishopTransform], curve: Callable[[Any], Any], /
) -> BishopTransform:
    """Construct a BishopTransform from a curve callable.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.curveframes as cxfc

    >>> def helix(tau: u.Q) -> u.Q:
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), t]), "m")

    >>> bt = cxfc.BishopTransform.from_(helix)
    >>> bt.location(u.Q(0.0, "s"))
    Q([1., 0., 0.], 'm')

    """
    return cls.from_curve(curve)


#####################################################################
# Frame


@final
class BishopFrame(AbstractParallelTransportFrame[FrameT]):
    """Bishop (rotation-minimising) curve-attached reference frame.

    A reference frame defined relative to a base frame by a `BishopTransform`.
    At each parameter value ``tau``, the frame is centred at the curve position
    with axes ``(T, U1, U2)`` obtained via parallel transport.

    Unlike `FrenetSerretFrame`, this frame is well-defined even at
    zero-curvature points.

    The evolution parameter ``tau`` is **not** stored on the frame; it is
    supplied at evaluation time via ``act(op, tau, x)``.

    Parameters
    ----------
    base_frame : AbstractReferenceFrame
        The ambient reference frame.
    xop : BishopTransform
        The tau-dependent rotation-minimising transform from ``base_frame`` to
        this frame.

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

    >>> b_frame = cxfc.BishopFrame.from_curve(cxf.Alice(), circle)
    >>> b_frame.base_frame
    Alice()

    >>> isinstance(b_frame.xop, cxfc.BishopTransform)
    True

    Get the frame transition operator and apply at tau=0:

    >>> op = cxf.frame_transition(cxf.Alice(), b_frame)
    >>> p = u.Q(jnp.array([1.0, 0.0, 0.0]), "km")
    >>> result = cxfm.act(op, u.Q(0.0, "s"), p)
    >>> jnp.allclose(result.value, jnp.array([0., 0., 0.]), atol=1e-5)
    Array(True, dtype=bool)

    """

    base_frame: FrameT
    xop: BishopTransform
    xop_inv: BishopTransform

    @classmethod
    def from_curve(
        cls,
        base_frame: FrameT,
        curve: Callable[[Any], Any],
        /,
        tau_unit: u.AbstractUnit | str = "s",
        *,
        tau_0: u.AbstractQuantity | None = None,
        initial_normal: Any | None = None,
    ) -> "BishopFrame[FrameT]":
        r"""Construct a BishopFrame from a base frame and curve.

        Parameters
        ----------
        base_frame : AbstractReferenceFrame
            The ambient reference frame.
        curve : Callable
            A function ``tau -> Quantity[float, (3,)]``.
        tau_unit : str, optional
            Unit of the curve parameter for differentiation.
        tau_0 : Quantity, optional
            Reference parameter.  Defaults to ``Q(0.0, tau_unit)``.
        initial_normal : array-like, optional
            Dimensionless 3-vector for $\mathbf{U}_{1,0}$.

        Returns
        -------
        BishopFrame

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

        >>> frame = cxfc.BishopFrame.from_curve(cxf.Alice(), circle)
        >>> frame.base_frame
        Alice()

        """
        xop = BishopTransform.from_curve(
            curve,
            tau_unit=tau_unit,
            tau_0=tau_0,
            initial_normal=initial_normal,
        )
        return cls(base_frame=base_frame, xop=xop, xop_inv=xop.inverse)
