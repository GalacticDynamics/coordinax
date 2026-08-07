r"""Abstract base classes for curve-attached reference frames.

This module defines the two abstract base classes on which the entire
``coordinaxs.curveframes`` package is built:

* `AbstractParallelTransportFrame` — a curve-attached orthonormal reference
  frame that integrates with the ``coordinax.frames`` frame-transition system.
* `AbstractCurveFrameBuilder` — the `equinox.Module` builder that a
  `coordinax.transforms.TimeDep` wraps: ``builder(tau)`` returns the
  rigid-body transform ``Translate(-gamma) | Rotate(R)`` at that parameter.

"""

__all__ = ("AbstractCurveFrameBuilder", "AbstractParallelTransportFrame")

import abc

from collections.abc import Callable
from jaxtyping import Array
from typing import Any
from typing_extensions import TypeVar

import equinox as eqx

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.transforms as cxfm
import unxt as u

FrameT = TypeVar(
    "FrameT", bound=cxf.AbstractReferenceFrame, default=cxf.AbstractReferenceFrame
)


class AbstractParallelTransportFrame(cxf.AbstractTransformedReferenceFrame[FrameT]):
    r"""Abstract base class for curve-attached orthonormal frames in 3-D.

    A parallel-transport frame attaches an orthonormal triad to each point of a
    smooth space curve $\boldsymbol{\gamma}(\tau)$.  Two concrete flavours are
    provided:

    * `FrenetSerretFrame` — axes are (T, N, B) from the Frenet--Serret
      apparatus; singular when curvature vanishes.
    * `BishopFrame` — axes are (T, U1, U2) obtained by parallel transport;
      well-defined for all regular curves.

    Inherits from `coordinax.frames.AbstractTransformedReferenceFrame` and
    therefore carries three fields:

    - ``base_frame`` — the ambient reference frame (e.g. ``Alice()``).
    - ``xop`` — the forward transform (base → curve frame), a
      `coordinax.transforms.TimeDep` wrapping an
      `AbstractCurveFrameBuilder`.
    - ``xop_inv`` — pre-computed inverse of ``xop`` (curve frame → base).

    Because this class is a  ``AbstractTransformedReferenceFrame``, the generic
    ``frame_transition`` dispatches apply automatically.  Concrete subclasses
    must be ``@final``.

    Notes
    -----
    The evolution parameter $\tau$ is **not** stored on the frame object.  It is
    supplied at evaluation time when the frame-transition operator is applied to
    coordinates via ``act(op, tau, x)``.

    Examples
    --------
    Concrete subclasses are used directly; see `FrenetSerretFrame` and
    `BishopFrame` for usage examples.

    """


# ============================================================================


class AbstractCurveFrameBuilder(eqx.Module):
    r"""ABC for curve-frame builders: ``tau -> Translate(-gamma) | Rotate(R)``.

    At each parameter value $\tau$, the built transform maps an ambient point
    $\mathbf{p}$ to curve-frame coordinates:

    $$
        \mathbf{p}' = R(\tau)\bigl(\mathbf{p}
                      - \boldsymbol{\gamma}(\tau)\bigr)
    $$

    decomposed into ``Translate(-gamma) | Rotate(R)``.

    Being an `equinox.Module`, every field is a pytree leaf: the curve's own
    parameters (when ``curve`` is itself an `equinox.Module`) and ``gamma`` are
    differentiable and vmappable.  A bare function passed as ``curve`` still
    works, but whatever it closes over is a trace-time constant.

    Fields
    ------
    curve : Callable
        The curve $\gamma \mapsto \boldsymbol{\gamma}(\gamma)$, mapping a
        parameter `Quantity` to a Cartesian 3-vector `Quantity`.
    tau_unit : AbstractUnit
        Physical unit of the curve parameter (e.g. ``"s"``).  Static: it selects
        the differentiation units, not a numeric value.
    gamma : Any, optional
        A *fixed* curve parameter.  When `None` (the default) $\tau$ itself is
        the curve parameter — the classic moving-frame usage.  When set, the
        frame sits at a fixed point of the curve and is $\tau$-independent: a
        frame *field* along the curve, differentiable and vmappable in
        ``gamma``.

    """

    curve: eqx.AbstractVar[Callable[[Any], Any]]
    tau_unit: eqx.AbstractVar[u.AbstractUnit]
    gamma: eqx.AbstractVar[Any]

    # ---------------------------------------------------------------

    def _param(self, tau: Any, /) -> Any:
        """Return the curve parameter: ``tau``, or the fixed ``gamma``."""
        return tau if self.gamma is None else self.gamma

    @abc.abstractmethod
    def rotation_matrix(self, tau: Any, /) -> Array:
        r"""Evaluate the $3 \times 3$ rotation matrix $R$ at ``tau``.

        The rows of $R$ are the frame vectors, e.g.
        $[\mathbf{T};\,\mathbf{N};\,\mathbf{B}]$ for Frenet--Serret.
        """
        raise NotImplementedError  # pragma: no cover

    def __call__(self, tau: Any, /) -> cxfm.Composed:
        r"""Build the transform at ``tau``: ``Translate(-gamma) | Rotate(R)``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.transforms as cxfm
        >>> import coordinaxs.curveframes as cxfc

        >>> def helix(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), t]), "m")

        >>> builder = cxfc.FrenetSerretBuilder(helix)
        >>> op = builder(u.Q(0.0, "s"))
        >>> isinstance(op, cxfm.Composed)
        True

        """
        cart = cxc.cart3d
        g = self._param(tau)
        translate = cxfm.Translate(cxc.cdict(-self.curve(g), cart), chart=cart)
        return translate | cxfm.Rotate(self.rotation_matrix(tau))

    # ---------------------------------------------------------------
    # Convenience accessors

    def location(self, tau: Any, /) -> Any:
        r"""Evaluate the curve position $\boldsymbol{\gamma}$ at ``tau``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def helix(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), t]), "m")

        >>> cxfc.FrenetSerretBuilder(helix).location(u.Q(0.0, "s"))
        Q([1., 0., 0.], 'm')

        """
        return self.curve(self._param(tau))

    def tangent(self, tau: Any, /) -> u.Q:
        r"""Return the unit tangent vector $\mathbf{T}$ (row 0 of R).

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinaxs.curveframes as cxfc

        >>> def circle(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> cxfc.FrenetSerretBuilder(circle).tangent(u.Q(0.0, "s"))
        Q([-0.,  1.,  0.], '')

        """
        R = self.rotation_matrix(tau.astype(float))
        return u.Q(R[0], "")
