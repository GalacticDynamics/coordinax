"""Register ``frame_transition`` dispatches for curve frames.

This module registers three ``plum.dispatch`` overloads of
{func}`frame_transition` that enable seamless integration of
`AbstractParallelTransportFrame` with the ``coordinax.frames`` frame-transition
system:

1. **To curve frame** — ``frame_transition(from_frame, to_curve_frame)``
   composes the transition to the curve frame's base frame with the curve-frame
   transform: ``(from -> base) | xop``.
2. **From curve frame** — ``frame_transition(from_curve_frame, to_frame)``
   composes the inverse curve-frame transform with the transition from the base
   frame: ``xop_inv | (base -> to)``.
3. **Between two curve frames** — ``frame_transition(from_curve, to_curve)``
   (precedence=1) composes both inverse and forward: ``xop_inv | (base1 ->
   base2) | xop``.

The higher precedence on dispatch (3) ensures it is selected when both arguments
are curve frames, rather than falling through to the more general dispatches (1)
or (2).
"""

__all__: tuple[str, ...] = ()

import plum

import coordinax.api.frames as api
import coordinax.frames as cxf
from coordinax.transforms import AbstractTransform

from .base import AbstractParallelTransportFrame


@plum.dispatch
def frame_transition(
    from_frame: cxf.AbstractReferenceFrame,
    to_frame: AbstractParallelTransportFrame,
) -> AbstractTransform:
    r"""Return the composite transform operator *to* a curve frame.

    Composes the transition from ``from_frame`` to the curve frame's
    ``base_frame`` with the curve-frame's forward transform ``xop``:

    $$ \mathcal{A} \to \mathcal{F}_\gamma
      = (\mathcal{A} \to \mathcal{B}) \circ
        (\mathcal{B} \to \mathcal{F}_\gamma)
    $$

    where $\mathcal{B}$ is ``to_frame.base_frame``.

    Parameters
    ----------
    from_frame : AbstractReferenceFrame
        The source (ambient) frame.
    to_frame : AbstractParallelTransportFrame
        The target curve-attached frame.

    Returns
    -------
    AbstractTransform
        The composed frame-transition operator.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.frames as cxf
    >>> import coordinax.transforms as cxt
    >>> import coordinax.curveframes as cxfc

    >>> def circle(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
    ...                           jnp.zeros_like(t)]), "km")

    >>> fs_frame = cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), circle)
    >>> op = cxf.frame_transition(cxf.Alice(), fs_frame)
    >>> isinstance(op, cxt.AbstractTransform)
    True

    """
    # Compose: (from_frame -> base_frame) then (base_frame -> curve_frame)
    return api.frame_transition(from_frame, to_frame.base_frame) | to_frame.xop  # ty: ignore[unsupported-operator]


@plum.dispatch
def frame_transition(  # noqa: F811
    from_frame: AbstractParallelTransportFrame,
    to_frame: cxf.AbstractReferenceFrame,
) -> AbstractTransform:
    r"""Return the composite transform operator *from* a curve frame.

    Composes the inverse curve-frame transform ``xop_inv`` with the transition
    from the base frame to ``to_frame``:

    $$ \mathcal{F}_\gamma \to \mathcal{A}
      = (\mathcal{F}_\gamma \to \mathcal{B}) \circ
        (\mathcal{B} \to \mathcal{A})
    $$

    Parameters
    ----------
    from_frame : AbstractParallelTransportFrame
        The source curve-attached frame.
    to_frame : AbstractReferenceFrame
        The target (ambient) frame.

    Returns
    -------
    AbstractTransform
        The composed frame-transition operator.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.frames as cxf
    >>> import coordinax.transforms as cxt
    >>> import coordinax.curveframes as cxfc

    >>> def circle(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
    ...                           jnp.zeros_like(t)]), "km")

    >>> fs_frame = cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), circle)
    >>> op = cxf.frame_transition(fs_frame, cxf.Alice())
    >>> isinstance(op, cxt.AbstractTransform)
    True

    """
    # Compose: (curve_frame -> base_frame) then (base_frame -> to_frame)
    return from_frame.xop_inv | api.frame_transition(from_frame.base_frame, to_frame)  # ty: ignore[unresolved-attribute]


@plum.dispatch(precedence=1)  # ty: ignore[no-matching-overload]
def frame_transition(  # noqa: F811
    from_frame: AbstractParallelTransportFrame,
    to_frame: AbstractParallelTransportFrame,
) -> AbstractTransform:
    r"""Return the composite transform operator between two curve frames.

    When both source and target are curve frames, the transition composes three
    stages:

    $$ \mathcal{F}_1 \to \mathcal{F}_2
      = (\mathcal{F}_1 \to \mathcal{B}_1) \circ
        (\mathcal{B}_1 \to \mathcal{B}_2) \circ (\mathcal{B}_2 \to
        \mathcal{F}_2)
    $$

    This dispatch has ``precedence=1`` so it takes priority over the more
    general to-curve-frame and from-curve-frame dispatches when both arguments
    are ``AbstractParallelTransportFrame``.

    Parameters
    ----------
    from_frame : AbstractParallelTransportFrame
        The source curve-attached frame.
    to_frame : AbstractParallelTransportFrame
        The target curve-attached frame.

    Returns
    -------
    AbstractTransform
        The composed frame-transition operator.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.frames as cxf
    >>> import coordinax.transforms as cxt
    >>> import coordinax.curveframes as cxfc

    >>> def circle(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t),
    ...                           jnp.zeros_like(t)]), "km")

    >>> fs1 = cxfc.FrenetSerretFrame.from_curve(cxf.Alice(), circle)
    >>> fs2 = cxfc.FrenetSerretFrame.from_curve(cxf.Alex(), circle)
    >>> op = cxf.frame_transition(fs1, fs2)
    >>> isinstance(op, cxt.AbstractTransform)
    True

    """
    # Compose: (curve1 -> base1) then (base1 -> base2) then (base2 -> curve2)
    return (
        from_frame.xop_inv  # ty: ignore[unresolved-attribute]
        | api.frame_transition(from_frame.base_frame, to_frame.base_frame)
        | to_frame.xop
    )
