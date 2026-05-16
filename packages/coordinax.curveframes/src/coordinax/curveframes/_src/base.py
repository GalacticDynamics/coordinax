r"""Abstract base classes for curve-attached reference frames.

This module defines the two abstract base classes on which the entire
``coordinax.curveframes`` package is built:

* `AbstractParallelTransportFrame` — a curve-attached orthonormal reference
  frame that integrates with the ``coordinax.frames`` frame-transition system.
* `AbstractParallelTransportTransform` — the corresponding rigid-body transform
  that decomposes into ``Translate(-gamma) | Rotate(R)``.

"""

__all__ = (
    "AbstractParallelTransportFrame",
    "AbstractParallelTransportTransform",
)

import abc

from collections.abc import Callable
from typing import Any
from typing_extensions import TypeVar, override

import equinox as eqx
import jax.numpy as jnp

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.transforms as cxt
import quaxed.numpy as qnp
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
    - ``xop`` — the forward transform (base → curve frame).
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


class AbstractParallelTransportTransform(cxt.AbstractCompositeTransform):
    r"""ABC for curve-attached rigid-body transforms (Translate | Rotate).

    At each parameter value $\tau$, the transform maps an ambient point
    $\mathbf{p}$ to curve-frame coordinates:

    $$
        \mathbf{p}' = R(\tau)\bigl(\mathbf{p}
                      - \boldsymbol{\gamma}(\tau)\bigr)
    $$

    Internally this is decomposed into a translation by $-\boldsymbol{\gamma}$
    followed by a rotation by $R$, i.e.  ``Translate(-gamma) | Rotate(R)``.

    Subclasses must be ``@final`` and implement:

    - ``_build_inverse()`` — construct the inverse as the same concrete type.
    - ``_rebuild_forward()`` — reconstruct the forward transform from the stored
      ``curve`` (used for double-inverse efficiency).

    Fields
    ------
    translate : Translate
        $\tau$-dependent translation whose delta callable evaluates to
        $-\boldsymbol{\gamma}(\tau)$.  Wrapped in a ``CDict`` keyed by
        ``cart3d``.
    rotate : Rotate
        $\tau$-dependent rotation whose ``R`` callable evaluates to the $3
        \times 3$ orthogonal matrix $[\mathbf{T};\,\mathbf{N};\,\mathbf{B}]$
        (rows) or the Bishop equivalent.
    curve : Callable
        The *original* constructing curve $\tau \mapsto
        \boldsymbol{\gamma}(\tau)$, kept so that double-inverse can cleanly
        reconstruct the forward transform without closure accumulation.
    tau_unit : AbstractUnit
        Physical unit of the curve parameter $\tau$ (e.g. ``"s"``).
    _is_forward : bool
        Sentinel flag.  ``True`` for forward transforms; ``False`` for inverses.
        Allows `.inverse` to detect the double-inverse case ``(F^{-1})^{-1}``
        and rebuild cleanly.

    Notes
    -----
    All fields with callable values are **lazy**: they are evaluated only when a
    concrete $\tau$ is supplied (typically inside ``act``).  This makes the
    transform a valid JAX PyTree (via Equinox) that is compatible with
    ``jax.jit`` and ``jax.vmap``.

    Examples
    --------
    Concrete subclasses are used directly; see `FrenetSerretTransform` and
    `BishopTransform` for usage examples.
    """

    translate: cxt.Translate
    rotate: cxt.Rotate

    curve: Callable[[Any], Any]
    """The original constructing curve."""

    tau_unit: u.AbstractUnit = eqx.field(static=True)
    """The unit of the curve parameter tau."""

    _is_forward: bool = eqx.field(static=True)
    """True for forward transform, False for inverse."""

    def __hash__(self) -> int:
        """Return an identity hash for JAX-bound method compatibility.

        `jax.jit(obj.method)` hashes the bound method, which includes `obj`.
        Structural hashing of Equinox modules may recurse into fields holding
        `Quantity` values with JAX arrays (e.g. Bishop's `tau_0`), which are not
        hashable. Using object identity keeps methods JIT-compatible while
        preserving immutable runtime behavior.
        """
        return object.__hash__(self)

    # ---------------------------------------------------------------
    # AbstractCompositeTransform interface

    @override
    @property
    def transforms(self) -> tuple[cxt.Translate, cxt.Rotate]:
        """Return the ordered pipeline of sub-transforms.

        The composite transform is always ``Translate(-gamma) | Rotate(R)``.
        The ``act`` dispatch iterates this tuple in order, applying each
        sub-transform to the intermediate result.

        Returns
        -------
        tuple[Translate, Rotate]
            Two-element tuple ``(translate, rotate)``.
        """
        return (self.translate, self.rotate)

    # ---------------------------------------------------------------
    # Convenience accessors

    @property
    def location(self) -> Callable[[u.AbstractQuantity], Any]:
        r"""Curve-location callable $\tau \mapsto \boldsymbol{\gamma}(\tau)$.

        For forward transforms, this is exactly the original ``curve`` callable,
        matching the spec requirement that ``location is curve``.

        For inverse transforms, ``location`` is derived from the stored
        translation field ``delta`` via

        $$ \boldsymbol{\gamma}_{\mathrm{inv}}(\tau) = -\,\Delta(\tau), $$

        where ``Translate`` acts as ``x -> x + Delta``.

        Returns
        -------
        Callable[[Quantity], Quantity]
            A callable that evaluates the transform's location at ``tau``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.curveframes as cxfc

        >>> def helix(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), t]), "m")

        >>> fs = cxfc.FrenetSerretTransform.from_curve(helix)
        >>> fs.location is fs.curve
        True

        >>> fs.location(u.Q(0.0, "s"))
        Q([1., 0., 0.], 'm')

        """
        if self._is_forward:
            return self.curve

        delta_fn = self.translate.delta

        def _location_from_delta(tau: u.AbstractQuantity, /) -> Any:
            delta = delta_fn(tau) if callable(delta_fn) else delta_fn  # ty: ignore[call-top-callable]
            return -qnp.stack(list(delta.values()), axis=-1)

        return _location_from_delta

    def _location_at(self, tau: u.AbstractQuantity, /) -> Any:
        r"""Evaluate the curve position $\boldsymbol{\gamma}(\tau)$.

        Parameters
        ----------
        tau : Quantity
            The evolution parameter value at which to evaluate.

        Returns
        -------
        Quantity
            The 3-D position vector along the curve.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.curveframes as cxfc

        >>> def helix(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), t]), "m")

        >>> fs = cxfc.FrenetSerretTransform.from_curve(helix)
        >>> fs.location(u.Q(0.0, "s"))
        Q([1., 0., 0.], 'm')

        """
        return self.location(tau)

    def _rotation_matrix(self, tau: u.AbstractQuantity, /) -> Any:
        r"""Evaluate the rotation matrix $R(\tau)$.

        For a forward transform the rows of $R$ are the frame vectors (e.g.
        $[\mathbf{T};\,\mathbf{N};\,\mathbf{B}]$).  For an inverse the rows are
        the *columns* of the forward $R$, i.e.  $R^{\mathsf{T}}$.

        Parameters
        ----------
        tau : Quantity
            The evolution parameter value.

        Returns
        -------
        Array, shape ``(3, 3)``
            Orthogonal rotation matrix.
        """
        R = self.rotate.R
        return R(tau) if callable(R) else R  # ty: ignore[call-top-callable]

    def tangent(self, tau: u.AbstractQuantity, /) -> Any:
        r"""Unit tangent vector $\mathbf{T}(\tau)$ (row 0 of R).

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

        >>> def circle(tau: u.Q) -> u.Q:
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> fs = cxfc.FrenetSerretTransform.from_curve(circle)
        >>> fs.tangent(u.Q(0.0, "s"))
        Q([-0.,  1.,  0.], '')

        """
        R = self._rotation_matrix(tau)
        return u.Q(R[0], "")

    # ---------------------------------------------------------------
    # Inverse

    @override
    @property
    def inverse(self) -> "AbstractParallelTransportTransform":
        r"""The inverse transform, preserving the concrete type.

        For a **forward** transform $\mathbf{p}' = R(\mathbf{p} -
        \boldsymbol{\gamma})$, the inverse is $\mathbf{p} = R^{\mathsf{T}}
        \mathbf{p}' + \boldsymbol{\gamma}$, which is re-expressed in the uniform
        pipeline form as ``Translate(-R gamma) | Rotate(R^T)``.

        For an **already-inverse** transform (``_is_forward is False``), calling
        ``.inverse`` triggers the *double-inverse* path: instead of wrapping
        closures around closures, the forward transform is cleanly
        **reconstructed** from the stored ``curve`` via ``_rebuild_forward()``.
        This guarantees a two-step cycle (forward ↔ inverse) with no
        closure-chain accumulation.

        Returns
        -------
        AbstractParallelTransportTransform
            The inverse transform, same concrete type as ``self``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.curveframes as cxfc

        >>> def circle(tau):
        ...     t = tau.ustrip("s")
        ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
        ...                           jnp.zeros_like(t)]), "m")

        >>> fs = cxfc.FrenetSerretTransform.from_curve(circle)
        >>> fs_inv = fs.inverse
        >>> isinstance(fs_inv, cxfc.FrenetSerretTransform)
        True

        Double-inverse reconstructs the forward transform:

        >>> fs_fwd = fs_inv.inverse
        >>> fs_fwd._is_forward
        True

        """
        # Double-inverse: rebuild the forward transform from scratch
        # to avoid accumulating closure layers.
        if not self._is_forward:
            return self._rebuild_forward()
        return self._build_inverse()

    @abc.abstractmethod
    def _build_inverse(self) -> "AbstractParallelTransportTransform":
        """Build the inverse transform as the same concrete type.

        Subclasses must construct a new instance with:

        - ``rotate``: ``Rotate(lambda tau: R(tau).T)``
        - ``translate``: ``Translate(lambda tau: cdict(-R(tau) @
          gamma(tau), cart3d))``
        - ``_is_forward = False``
        - All extra fields (e.g. Bishop's ``tau_0``, ``initial_normal``)
        """
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def _rebuild_forward(self) -> "AbstractParallelTransportTransform":
        """Reconstruct the forward transform from the stored curve.

        Called when ``.inverse`` is invoked on an already-inverse instance,
        avoiding closure accumulation.
        """
        raise NotImplementedError  # pragma: no cover

    # ---------------------------------------------------------------
    # Inverse-building helpers (shared logic)

    def _make_inverse_rotate(self) -> cxt.Rotate:
        r"""Build the inverse rotation: $R^{\mathsf{T}}$.

        Since $R \in SO(3)$, its inverse is simply its transpose.  The returned
        ``Rotate`` wraps a callable that transposes the forward rotation matrix
        at each $\tau$.

        """
        R_fn = self.rotate.R
        inv_R = (  # noqa: E731
            lambda tau: jnp.swapaxes(R_fn(tau), -2, -1)  # ty: ignore[call-top-callable]
            if callable(R_fn)
            else jnp.swapaxes(R_fn, -2, -1)
        )
        return cxt.Rotate(inv_R)

    def _make_inverse_translate(self) -> cxt.Translate:
        r"""Build the inverse translation: $-R(\tau) \boldsymbol{\gamma}(\tau)$.

        For the inverse transform the translation step shifts by
        $-R\boldsymbol{\gamma}$ (note the sign: the forward translate is
        $-\boldsymbol{\gamma}$, so the inverse translate is
        $-R\boldsymbol{\gamma}$ which, after the inverse rotation
        $R^{\mathsf{T}}$, recovers $+\boldsymbol{\gamma}$).

        """
        R_fn = self.rotate.R
        curve = self.curve
        cart = cxc.cart3d

        def inv_delta(tau: Any, /) -> Any:
            # Evaluate R and gamma at the requested tau, then compute
            # the rotated translation vector  R @ gamma.
            R = R_fn(tau) if callable(R_fn) else R_fn  # ty: ignore[call-top-callable]
            gamma = curve(tau)
            R_gamma = qnp.einsum("ij,...j->...i", R, gamma)
            return cxc.cdict(R_gamma, cart)

        return cxt.Translate(inv_delta, chart=cart)
