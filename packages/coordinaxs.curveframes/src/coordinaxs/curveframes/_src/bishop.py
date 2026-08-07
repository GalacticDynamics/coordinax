r"""Bishop (rotation-minimising) curve-frame data types.

This module provides the concrete implementations of the Bishop
(parallel-transport, rotation-minimising) curve-frame apparatus:

* `BishopBuilder` — an `equinox.Module` mapping $\tau$ to the rigid-body
  transform ``Translate(-\gamma) | Rotate([T; U1; U2])``.
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
* **Lazy evaluation** — the ODE is solved only when a concrete $\tau$ is
  requested, i.e. when the `TimeDep` family is materialized.
* **Auto initial normal** — when no ``initial_normal`` is supplied, one is
  chosen automatically via Gram--Schmidt against the tangent at $\tau_0$.

See Also
--------
coordinaxs.curveframes._src.frenetserret : Frenet--Serret frame.
coordinaxs.curveframes._src.base : Abstract base classes.

"""

__all__ = ("BishopBuilder", "BishopFrame")

from collections.abc import Callable
from jaxtyping import Array
from typing import Any, cast, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

import coordinax.transforms as cxfm
import unxt as u

from .base import AbstractCurveFrameBuilder, AbstractParallelTransportFrame, FrameT
from .frenetserret import _normalize

_MSG_PARALLEL_NORMAL = (
    "`initial_normal` is parallel to the tangent at tau_0; it has no component "
    "in the normal plane. Pass a vector that is not along the tangent, or leave "
    "it `None` to have one chosen automatically."
)


def _orthonormalize(v: Any, T0_val: Any) -> Any:
    r"""Gram--Schmidt ``v`` against the unit tangent, then normalise.

    Raises (via `equinox.error_if`, so it also fires under ``jit``) when ``v``
    is parallel to ``T0_val`` and the rejection vanishes.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from coordinaxs.curveframes._src.bishop import _orthonormalize

    >>> T0 = jnp.array([1.0, 0.0, 0.0])
    >>> _orthonormalize(jnp.array([2.0, 3.0, 0.0]), T0)
    Array([0., 1., 0.], dtype=float64)

    """
    w = v - jnp.dot(v, T0_val) * T0_val
    norm = jnp.linalg.norm(w)
    w = eqx.error_if(w, norm < 1e-12, _MSG_PARALLEL_NORMAL)
    return w / norm


def _auto_initial_normal(T0_val: Any) -> Any:
    r"""Choose an initial unit normal to the unit tangent $\mathbf{T}_0$.

    Gram--Schmidt against the standard basis vector least aligned with
    $\mathbf{T}_0$, which keeps the rejection well-conditioned even when
    $\mathbf{T}_0$ nearly coincides with a coordinate axis.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from coordinaxs.curveframes._src.bishop import _auto_initial_normal

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
    return _orthonormalize(e_k, T0_val)


@final
class BishopBuilder(AbstractCurveFrameBuilder):
    r"""Bishop (rotation-minimising) frame family along a curve.

    The Bishop frame attaches an orthonormal triad $(\mathbf{T}, \mathbf{U}_1,
    \mathbf{U}_2)$ to each point of a smooth space curve $\gamma(\tau)$ via
    parallel transport:

    - $\mathbf{T}$ (tangent): unit tangent vector $\gamma'/\|\gamma'\|$
    - $\mathbf{U}_1$ (normal 1): parallel-transported first normal
    - $\mathbf{U}_2$ (normal 2): $\mathbf{T} \times \mathbf{U}_1$

    Unlike the Frenet-Serret frame, the Bishop frame is well-defined even when
    the curvature vanishes ($\kappa = 0$).

    Calling the builder at $\tau$ returns ``Translate(-gamma) | Rotate(R)`` with
    $R = [\mathbf{T};\,\mathbf{U}_1;\,\mathbf{U}_2]$.

    Parameters
    ----------
    curve : Callable
        A function ``tau -> Quantity[float, (3,)]``.  Make it an
        `equinox.Module` for differentiable curve parameters.
    tau_unit : AbstractUnit or str, optional
        Unit of the curve parameter.  Defaults to ``"s"``.
    gamma : optional
        A fixed curve parameter; see `AbstractCurveFrameBuilder`.
    tau_0 : Quantity, optional
        Reference parameter where the initial frame is defined.  Defaults to
        ``Q(0.0, tau_unit)``.
    initial_normal : array-like, optional
        Dimensionless 3-vector for $\mathbf{U}_{1,0}$.  When `None`,
        auto-chosen via Gram--Schmidt against the tangent at ``tau_0``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinaxs.curveframes as cxfc

    Define a helix:

    >>> def helix(tau: u.Q) -> u.Q:
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), t]), "m")

    >>> bt = cxfc.BishopBuilder(helix)
    >>> bt.location(u.Q(0.0, "s"))
    Q([1., 0., 0.], 'm')

    The Bishop frame works on a straight line (where Frenet-Serret is singular):

    >>> def line(tau: u.Q) -> u.Q:
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([t, jnp.zeros_like(t),
    ...                           jnp.zeros_like(t)]), "m")

    >>> U1 = cxfc.BishopBuilder(line).normal1(u.Q(5.0, "s"))
    >>> jnp.sqrt(jnp.sum(U1.value**2))
    Array(1., dtype=float64)

    """

    curve: Callable[[Any], Any]
    """The constructing curve."""

    tau_unit: u.AbstractUnit = eqx.field(  # ty: ignore[invalid-assignment]
        default=u.unit("s"), static=True, converter=u.unit
    )
    """The unit of the curve parameter tau."""

    gamma: Any = None
    """Optional fixed curve parameter (a leaf); `None` means "use tau"."""

    tau_0: u.AbstractQuantity | None = None
    """Reference parameter value where the initial frame is defined (a leaf).

    `None` is resolved to ``Q(0.0, tau_unit)`` by ``__post_init__``.
    """

    initial_normal: Any = None
    """Initial U1 vector at tau_0 (dimensionless jax array, or None for auto)."""

    def __post_init__(self) -> None:
        """Resolve a `None` ``tau_0`` to zero in ``tau_unit`` (a pytree leaf)."""
        if self.tau_0 is None:
            self.tau_0 = u.Q(0.0, self.tau_unit)

    # ---------------------------------------------------------------

    def _tangent_at(self, g: Any, /) -> u.AbstractQuantity:
        r"""Compute unit tangent $\mathbf{T} = \gamma'/\|\gamma'\|$."""
        dcurve = u.experimental.jacfwd(self.curve, units=(self.tau_unit,))
        return _normalize(dcurve(g.astype(float)))

    def _solve_U1(self, g: Any, /) -> Array:
        r"""Compute $\mathbf{U}_1$ via ODE integration from $\tau_0$.

        Solves the parallel-transport ODE $d\mathbf{U}_1/d\tau = -(\mathbf{U}_1
        \cdot \mathbf{T}')\,\mathbf{T}$ using
        ``jax.experimental.ode.odeint``.  When the parameter equals $\tau_0$,
        the ODE is skipped via ``jax.lax.cond`` (identity path).
        """
        tau_unit = self.tau_unit
        tau_0 = cast("u.AbstractQuantity", self.tau_0)

        T0_val = self._tangent_at(tau_0).value
        if self.initial_normal is not None:
            # A supplied vector is NOT trusted to be unit or normal-plane: the
            # transport ODE conserves any error in it forever, so R would not
            # be a rotation.  Gram--Schmidt it exactly as the auto path does.
            U1_0_val = _orthonormalize(
                jnp.asarray(self.initial_normal, dtype=float), T0_val
            )
        else:
            U1_0_val = _auto_initial_normal(T0_val)

        # Pre-compute dT/dtau as a callable.  This avoids nesting AD inside
        # the ODE right-hand-side, which would be both slower and harder
        # for JAX to trace.
        dTangent_fn = u.experimental.jacfwd(self._tangent_at, units=(tau_unit,))

        tau_val = g.ustrip(tau_unit)
        tau_0_val = tau_0.ustrip(tau_unit)

        def ode_rhs(U1_flat: Any, t_scalar: Any) -> Any:
            """Right-hand side of the parallel-transport ODE."""
            t_q = u.Q(t_scalar, tau_unit)
            T_val = self._tangent_at(t_q).value
            dT_val = dTangent_fn(t_q).value
            # Project U1 onto dT, negate, then scale by T.
            return -jnp.dot(U1_flat, dT_val) * T_val

        # Use lax.cond to branch: when tau == tau_0, return initial
        # normal directly (avoids zero-length ODE integration).
        needs_ode = jnp.abs(tau_val - tau_0_val) > 0.0

        def _solve(_: Any) -> Any:
            # `odeint` integrates FORWARD only: a decreasing `t_span` yields
            # NaN.  Integrate the reversed field over s in [0, |dtau|] instead,
            # which is the same solution for either sign of dtau.
            dtau = tau_val - tau_0_val
            sgn = jnp.sign(dtau)
            s_span = jnp.stack([jnp.zeros_like(dtau), jnp.abs(dtau)])
            result = odeint(
                lambda y, s: sgn * ode_rhs(y, tau_0_val + sgn * s), U1_0_val, s_span
            )
            return result[-1]  # solution at tau

        def _identity(_: Any) -> Any:
            return U1_0_val

        U1_val = jax.lax.cond(needs_ode, _solve, _identity, None)
        # Re-normalise for numerical safety.
        return U1_val / jnp.linalg.norm(U1_val)

    def rotation_matrix(self, tau: Any, /) -> Array:
        r"""Compute the rotation $R = [T;\,U_1;\,U_2]$.

        Steps:

        1. Evaluate the tangent $\mathbf{T}$ from the first derivative.
        2. Solve the parallel-transport ODE for $\mathbf{U}_1$.
        3. Cross product: $\mathbf{U}_2 = \mathbf{T} \times \mathbf{U}_1$.
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

        >>> R = cxfc.BishopBuilder(circle).rotation_matrix(u.Q(0.0, "s"))
        >>> bool(jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-6))
        True

        """
        g = self._param(tau)
        T_val = self._tangent_at(g).value
        U1_val = self._solve_U1(g)
        U2_val = jnp.cross(T_val, U1_val)
        return jnp.stack([T_val, U1_val, U2_val])

    # ---------------------------------------------------------------
    # Convenience accessors (location inherited from the ABC)

    def tangent(self, tau: Any, /) -> u.Q:
        r"""Return the unit tangent vector $\mathbf{T}(\tau)$ (row 0 of R).

        Overrides the base implementation, which would take row 0 of the full
        rotation matrix — and so pay for the parallel-transport ODE solve that
        only $\mathbf{U}_1$ needs. The tangent is just the normalised first
        derivative; the value is identical.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> cxfc.BishopBuilder(circle).tangent(u.Q(0.0, "s"))
        Q([-0.,  1.,  0.], '')

        """
        return u.Q(self._tangent_at(self._param(tau)).value, "")

    def normal1(self, tau: Any, /) -> u.Q:
        r"""First parallel-transported normal $\mathbf{U}_1(\tau)$ (row 1 of R).

        This vector is obtained by solving the parallel-transport ODE from the
        reference parameter $\tau_0$.  It is perpendicular to the tangent and
        rotation-minimising: the angular velocity of the frame about the tangent
        is zero.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> U1 = cxfc.BishopBuilder(circle).normal1(u.Q(0.0, "s"))
        >>> float(jnp.linalg.norm(U1.value))
        1.0

        """
        return u.Q(self.rotation_matrix(tau)[1], "")

    def normal2(self, tau: Any, /) -> u.Q:
        r"""Second normal $\mathbf{U}_2(\tau) = \mathbf{T} \times \mathbf{U}_1$.

        The second normal completes the right-handed orthonormal triad.  It is
        computed as the cross product of the tangent and the first normal, so it
        is automatically perpendicular to both.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> U2 = cxfc.BishopBuilder(circle).normal2(u.Q(0.0, "s"))
        >>> float(jnp.linalg.norm(U2.value))
        1.0

        """
        return u.Q(self.rotation_matrix(tau)[2], "")


#####################################################################
# Frame


@final
class BishopFrame(AbstractParallelTransportFrame[FrameT]):
    """Bishop (rotation-minimising) curve-attached reference frame.

    A reference frame defined relative to a base frame by a
    `coordinax.transforms.TimeDep` wrapping a `BishopBuilder`.  At each
    parameter value ``tau``, the frame is centred at the curve position with
    axes ``(T, U1, U2)`` obtained via parallel transport.

    Unlike `FrenetSerretFrame`, this frame is well-defined even at
    zero-curvature points.

    The evolution parameter ``tau`` is **not** stored on the frame; it is
    supplied at evaluation time via ``act(op, tau, x)``.

    Parameters
    ----------
    base_frame : AbstractReferenceFrame
        The ambient reference frame.
    xop : TimeDep
        The tau-dependent rotation-minimising transform from ``base_frame`` to
        this frame.
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

    >>> b_frame = cxfc.BishopFrame.from_curve(cxf.Alice(), circle)
    >>> b_frame.base_frame
    Alice()

    >>> isinstance(b_frame.xop.builder, cxfc.BishopBuilder)
    True

    Get the frame transition operator and apply at tau=0:

    >>> op = cxf.frame_transition(cxf.Alice(), b_frame)
    >>> p = u.Q(jnp.array([1.0, 0.0, 0.0]), "km")
    >>> result = cxfm.act(op, u.Q(0.0, "s"), p)
    >>> jnp.allclose(result.value, jnp.array([0., 0., 0.]), atol=1e-5)
    Array(True, dtype=bool)

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
        tau_unit: u.AbstractUnit | str = "s",
        *,
        gamma: Any = None,
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
        gamma : optional
            A fixed curve parameter; when given the frame is a fixed frame
            *field* along the curve rather than a moving frame.
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
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "km")

        >>> frame = cxfc.BishopFrame.from_curve(cxf.Alice(), circle)
        >>> frame.base_frame
        Alice()

        """
        builder = BishopBuilder(curve, tau_unit, gamma, tau_0, initial_normal)
        xop = cxfm.TimeDep(builder)
        return cls(base_frame=base_frame, xop=xop, xop_inv=xop.inverse)
