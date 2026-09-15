r"""Signed planar curve-frame data types.

This module provides the concrete implementations of the signed planar
curve-frame apparatus:

* `SignedPlanarBuilder` — an `equinox.Module` mapping $\tau$ to the rigid-body
  transform ``Translate(-\gamma) | Rotate([T; N; B])``.
* `SignedPlanarFrame` — a curve-attached reference frame whose axes are
  $(\mathbf{T}, \mathbf{N}, \mathbf{B})$ with $\mathbf{N}$ a quarter turn to
  the left of $\mathbf{T}$ within a fixed plane.

Unlike the Frenet--Serret frame, this one is **defined where the curvature
vanishes** ($\kappa = 0$) — every straight segment and every inflection —
because the normal is not derived from $\boldsymbol{\gamma}''$ at all.  It
comes from the ambient plane:

$$ \mathbf{T} = \widehat{\boldsymbol{\gamma}'}, \qquad
   \mathbf{N} = \widehat{\hat{n} \times \mathbf{T}}, \qquad
   \mathbf{B} = \mathbf{T} \times \mathbf{N},
$$

so no vanishing vector is ever normalised.  The price is that the curve must
be **planar**: $\hat{n}$ is what fixes the gauge, and a curve that leaves the
plane normal to $\hat{n}$ is refused rather than silently projected.

At an inflection the normal *plane* is perfectly well-defined; what
degenerates for Frenet--Serret is only the choice of basis within it.  That is
a gauge degeneracy, and this type resolves it by taking the gauge from the
ambient plane.  `BishopBuilder` resolves the same degeneracy by transport,
which costs an initial normal and an ODE solve but works in any dimension.

Both classes are ``@final`` (no further subclassing).

See Also
--------
coordinaxs.curveframes._src.frenetserret : Frenet--Serret frame.
coordinaxs.curveframes._src.bishop : Bishop (rotation-minimising) frame.
coordinaxs.curveframes._src.base : Abstract base classes.

"""

__all__ = ("SignedPlanarBuilder", "SignedPlanarFrame")

from collections.abc import Callable
from jaxtyping import Array
from typing import Any, cast, final

import equinox as eqx
import jax.numpy as jnp

import coordinax.transforms as cxfm
import quaxed.numpy as qnp
import unxt as u
from unxt.quantity import AllowValue

from .base import (
    AbstractCurveFrameBuilder,
    AbstractParallelTransportFrame,
    FrameT,
    unit_or_none,
)
from .bishop import _float
from .frenetserret import _normalize

_MSG_NOT_PLANAR = (
    "the signed planar frame needs a curve lying in the plane normal to "
    "`plane_normal`, and this one leaves it: the tangent has a component "
    "along `plane_normal` far above the working precision. If the curve is "
    "planar but in some other plane, name that plane with `plane_normal=`. "
    "If it is genuinely not planar -- a helix, say -- then no plane can fix "
    "the gauge: use `BishopBuilder` instead, whose rotation-minimising frame "
    "is defined for every regular curve, in or out of a plane."
)

_MSG_DEGENERATE_PLANE_NORMAL = (
    "`plane_normal` has zero length, so it names no plane and would "
    "normalise to NaN, poisoning the whole triad. Pass a nonzero 3-vector, "
    "or leave it `None` to take the z-axis."
)


def _planar_tol(x: Any, /) -> Array:
    r"""Out-of-plane tolerance: $\sqrt{\varepsilon}$ of the working dtype.

    About ``1.5e-8`` in f64 and ``3.4e-4`` in f32.

    Deliberately *not* the hardcoded ``1e-12`` that `bishop._orthonormalize`
    and `frenetserret.rotation_matrix` use. Those thresholds sit on a
    *relative* rejection magnitude, where one fixed small number is meaningful
    in either precision. This one sits on an absolute angle sine, whose
    achievable floor is set by the dtype: a hardcoded ``1e-12`` would reject
    every f32 curve, and a hardcoded ``1e-4`` would be needlessly blind in f64.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from coordinaxs.curveframes._src.signedplanar import _planar_tol

    >>> bool(_planar_tol(jnp.float64(1.0)) < 1e-7)
    True

    """
    return jnp.sqrt(jnp.finfo(jnp.result_type(x, float)).eps)


def _check_planar(t_vec: Any, n_hat: Array, /) -> Any:
    r"""Raise (under ``jit`` too) when the tangent leaves the plane.

    Returns the *checked* tangent, which callers must then use: an
    `equinox.error_if` whose result is discarded is dead code, and JAX will
    eliminate it along with the check.

    The orthonormality defect is second order in the out-of-plane angle --
    $\|\hat{n} \times \mathbf{T}\|^2 = 1 - (\hat{n}\cdot\mathbf{T})^2$, so a
    $10^{-6}$ tilt costs $5\times10^{-13}$ in the norm. So this guard is not
    protecting $R$ from being a non-rotation; it is protecting the *claim*
    that the frame is the left-normal frame of the plane the caller named.

    ``~(x < tol)`` rather than ``x >= tol``, for the reason
    `frenetserret.rotation_matrix`'s guard gives: NaN is False under both, so
    the ``>=`` form would pass an already-NaN tangent straight through the
    guard that exists to stop it.
    """
    dot = cast("Array", u.ustrip(AllowValue, "", qnp.sum(t_vec * n_hat)))
    return eqx.error_if(t_vec, ~(jnp.abs(dot) < _planar_tol(dot)), _MSG_NOT_PLANAR)


@final
class SignedPlanarBuilder(AbstractCurveFrameBuilder):
    r"""Signed planar frame family along a curve.

    Attaches an orthonormal triad $(\mathbf{T}, \mathbf{N}, \mathbf{B})$ to
    each point of a smooth **planar** curve $\gamma(\tau)$:

    - $\mathbf{T}$ (tangent): $\gamma'/\|\gamma'\|$
    - $\mathbf{N}$ (normal): $\hat{n} \times \mathbf{T}$, normalised — a
      quarter turn to the *left* of travel, seen from $+\hat{n}$
    - $\mathbf{B}$ (binormal): $\mathbf{T} \times \mathbf{N}$, which is
      $\hat{n}$ itself on a planar curve

    Contrast `FrenetSerretBuilder`, whose $\mathbf{N}$ points at the centre of
    curvature and is undefined where the curvature vanishes.  The two are
    **not** interchangeable: where both are defined they agree up to a sign,
    and that sign flips at every inflection.

    Parameters
    ----------
    curve : Callable
        A function ``tau -> Quantity[float, (3,)]`` representing a smooth
        planar curve.  Make it an `equinox.Module` for differentiable curve
        parameters; a bare function's captures are trace-time constants.
    tau_unit : AbstractUnit or str, optional
        Unit of the curve parameter; `None` (the default) reads it off the
        parameter the builder is called with.  See
        `AbstractCurveFrameBuilder`.
    station : optional
        A fixed station along the curve; see `AbstractCurveFrameBuilder`.
    plane_normal : array-like, optional
        Dimensionless 3-vector normal to the plane the curve lies in; it need
        not be normalised.  `None` (the default) takes the z-axis.  This is
        the *gauge*: it is what makes the frame defined at an inflection, and
        it is an input rather than something derived from the curve, because
        anything derived from $\gamma''$ vanishes exactly where this frame is
        wanted.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinaxs.curveframes as cxfc

    A cubic with an inflection at the origin, where Frenet--Serret is
    undefined:

    >>> def cubic(tau: u.Q) -> u.Q:
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([t, t**3, jnp.zeros_like(t)]), "m")

    >>> sp = cxfc.SignedPlanarBuilder(cubic, "s")
    >>> sp.location(u.Q(0.0, "s"))
    Q([0., 0., 0.], 'm')

    """

    curve: Callable[[Any], Any]
    """The constructing curve."""

    tau_unit: u.AbstractUnit | None = eqx.field(
        default=None, static=True, converter=unit_or_none
    )
    """The unit of the curve parameter tau."""

    station: Any = None
    """Optional fixed station along the curve (a leaf); `None` means "use tau"."""

    plane_normal: Any = None
    """Normal of the plane the curve lies in (a leaf); `None` means the z-axis."""

    def _plane_normal(self, dtype: Any, /) -> Array:
        """Resolve `plane_normal` to a *unit* 3-vector in ``dtype``.

        Taking the dtype from the caller rather than defaulting it keeps an
        f32 curve in f32: a bare ``jnp.array([0., 0., 1.])`` is f64 under
        ``jax_enable_x64`` and would silently widen the whole triad, and with
        it the tolerance `_planar_tol` picks.
        """
        if self.plane_normal is None:
            return jnp.array([0.0, 0.0, 1.0], dtype=dtype)
        n = _float(self.plane_normal).astype(dtype)
        norm = jnp.linalg.norm(n)
        # `~(norm > 0)`, not `norm <= 0`: NaN is False for both.
        n = eqx.error_if(n, ~(norm > 0), _MSG_DEGENERATE_PLANE_NORMAL)
        return n / norm

    def rotation_matrix(self, tau: Any, /) -> Array:
        r"""Compute the full rotation matrix $R = [T; N; B]$.

        Steps:

        1. Evaluate the tangent $\mathbf{T} = \gamma'/\|\gamma'\|$.
        2. Check the curve has not left the plane (see `_check_planar`).
        3. Rotate: $\mathbf{N} = \widehat{\hat{n} \times \mathbf{T}}$.
        4. Cross product: $\mathbf{B} = \mathbf{T} \times \mathbf{N}$.
        5. Stack rows into a $3 \times 3$ matrix.

        $\boldsymbol{\gamma}''$ is never evaluated, so this costs one
        `unxt.experimental.jacfwd` where `FrenetSerretBuilder` costs two.
        Only `signed_curvature` pays for the second derivative.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> cxfc.SignedPlanarBuilder(circle, "s").rotation_matrix(
        ...     u.Q(0.0, "s")).round(3)
        Array([[-0.,  1.,  0.],
               [-1., -0.,  0.],
               [ 0.,  0.,  1.]], dtype=float64)

        """
        # For a two-argument curve `tau` is the time: the apparatus is that
        # of the time slice, at the pinned station. See `_resolve`.
        b, p = self._resolve(tau)

        g, tau_unit = b._param(p)
        dcurve = u.experimental.jacfwd(b.curve, units=(tau_unit,))
        t_vec = _normalize(dcurve(g.astype(float)))

        n_hat = b._plane_normal(jnp.result_type(t_vec.value, float))
        # Use the *returned* tangent: an `error_if` whose result is dropped
        # is dead code, and the check goes with it.
        t_vec = _check_planar(t_vec, n_hat)

        # A quarter turn within the plane. `n_hat x T` has norm
        # `sqrt(1 - (n_hat.T)^2)`, which the guard above has just pinned near
        # 1, so this normalisation divides by ~1 and never by ~0 -- which is
        # the whole point of the type.
        n_vec = _normalize(qnp.cross(n_hat, t_vec))
        b_vec = qnp.cross(t_vec, n_vec)

        # ``Rotate`` expects a bare numerical array, not a ``Quantity``.
        return qnp.stack([t_vec, n_vec, b_vec]).value  # ty: ignore[unresolved-attribute]

    def normal(self, tau: Any, /) -> u.Q:
        r"""Return the unit normal $\mathbf{N}(\tau)$ (row 1 of R).

        A quarter turn to the left of travel, seen from $+\hat{n}$.  Unlike
        the Frenet--Serret normal it is continuous through an inflection, and
        it is defined on a straight line.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def line(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([t, jnp.zeros_like(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> cxfc.SignedPlanarBuilder(line, "s").normal(u.Q(5.0, "s"))
        Q([0., 1., 0.], '')

        """
        return u.Q(self.rotation_matrix(tau)[1], "")

    def binormal(self, tau: Any, /) -> u.Q:
        r"""Return the unit binormal $\mathbf{B}(\tau)$ (row 2 of R).

        On a planar curve this *is* the plane normal $\hat{n}$, at every
        parameter — the frame does not twist out of the plane.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> cxfc.SignedPlanarBuilder(circle, "s").binormal(u.Q(1.3, "s"))
        Q([0., 0., 1.], '')

        """
        return u.Q(self.rotation_matrix(tau)[2], "")

    def signed_curvature(self, tau: Any, /) -> u.Q:
        r"""Return the signed curvature $\kappa_s(\tau)$.

        $$ \kappa_s = \frac{(\boldsymbol{\gamma}' \times
           \boldsymbol{\gamma}'') \cdot \hat{n}}
           {\|\boldsymbol{\gamma}'\|^3} $$

        Unlike the Frenet--Serret $\kappa \ge 0$, this is signed: positive
        where the curve turns towards $\mathbf{N}$ (left, seen from
        $+\hat{n}$), negative where it turns away, and zero at an inflection
        — through which it passes smoothly rather than being undefined.

        The sign is fixed by $d\mathbf{T}/ds = \kappa_s \mathbf{N}$, and on a
        counter-clockwise circle it agrees with Frenet's $\kappa$.

        Returns a `Quantity` of dimension 1/length.  This is the only
        curvature accessor in the package; `FrenetSerretBuilder` has none.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        A cubic, at and around its inflection:

        >>> def cubic(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([t, t**3, jnp.zeros_like(t)]), "km")

        >>> sp = cxfc.SignedPlanarBuilder(cubic, "s")
        >>> [float(sp.signed_curvature(u.Q(t, "s")).ustrip("1/km").round(6))
        ...  for t in (-1e-3, 0.0, 1e-3)]
        [-0.006, 0.0, 0.006]

        """
        b, p = self._resolve(tau)

        g, tau_unit = b._param(p)
        g = g.astype(float)
        dcurve = u.experimental.jacfwd(b.curve, units=(tau_unit,))
        d2curve = u.experimental.jacfwd(dcurve, units=(tau_unit,))
        dp = dcurve(g)
        d2p = d2curve(g)

        n_hat = b._plane_normal(jnp.result_type(dp.value, float))
        # Threading the checked tangent through the numerator is what keeps
        # the guard alive: an `error_if` whose result is dropped is dead code.
        # It also cancels one power of |gamma'|, since
        # (T x gamma'') . n / |gamma'|^2 == (gamma' x gamma'') . n / |gamma'|^3.
        t_vec = _check_planar(_normalize(dp), n_hat)

        return qnp.sum(qnp.cross(t_vec, d2p) * n_hat) / qnp.sum(dp**2)


#####################################################################
# Frame


@final
class SignedPlanarFrame(AbstractParallelTransportFrame[FrameT]):
    """Signed planar curve-attached reference frame.

    A reference frame defined relative to a base frame by a
    `coordinax.transforms.TimeDep` wrapping a `SignedPlanarBuilder`.  At each
    parameter value ``tau``, the frame is centred at the curve position with
    axes ``(T, N, B)``, where ``N`` is a quarter turn to the left of travel
    within the curve's plane.

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
    >>> import coordinaxs.curveframes as cxfc

    A cubic, whose inflection at the origin no frame here has trouble with:

    >>> def cubic(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([t, t**3, jnp.zeros_like(t)]), "km")

    >>> frame = cxfc.SignedPlanarFrame.from_curve(cxf.Alice(), cubic, "s")
    >>> frame.base_frame
    Alice()

    >>> isinstance(frame.xop.builder, cxfc.SignedPlanarBuilder)
    True

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
        plane_normal: Any = None,
    ) -> "SignedPlanarFrame[FrameT]":
        """Construct a SignedPlanarFrame from a base frame and curve.

        Parameters
        ----------
        base_frame : AbstractReferenceFrame
            The ambient reference frame.
        curve : Callable
            A function ``tau -> Quantity[float, (3,)]`` representing a smooth
            planar curve.
        tau_unit : AbstractUnit or str, optional
            Unit of the curve parameter for differentiation.  `None` (the
            default) reads it off the parameter the frame is evaluated at.
        station : optional
            A fixed station along the curve; when given the frame is a fixed
            frame *field* along the curve rather than a moving frame.
        plane_normal : array-like, optional
            Normal of the plane the curve lies in; `None` takes the z-axis.
            See `SignedPlanarBuilder`.

        Returns
        -------
        SignedPlanarFrame
            A frame attached to the curve, relative to ``base_frame``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.frames as cxf
        >>> import coordinaxs.curveframes as cxfc

        >>> def line(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([t, jnp.zeros_like(t),
        ...                           jnp.zeros_like(t)]), "km")

        A straight line, where Frenet--Serret has no frame at all:

        >>> frame = cxfc.SignedPlanarFrame.from_curve(cxf.Alice(), line, "s")
        >>> frame.base_frame
        Alice()

        """
        builder = SignedPlanarBuilder(curve, tau_unit, station, plane_normal)
        xop = cxfm.TimeDep(builder)
        return cls(base_frame=base_frame, xop=xop, xop_inv=xop.inverse)
