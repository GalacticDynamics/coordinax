"""Representations."""

__all__: tuple[str, ...] = ("act", "compose", "prolong", "pushforward")

from typing import Any

import plum


@plum.dispatch.abstract
def act(*args: Any, **kwargs: Any) -> Any:
    r"""Apply a transform action to coordinates.

    This is the core dispatch function for transform application. Each transform
    type registers its own implementation via multiple dispatch. Transforms act
    on various input types (Array, Quantity, Vector, CDict) according to their
    semantics.

    Mathematical Definition:

    For a transform $\mathcal{T}$ parameterized by $\tau$, this computes:

    $$ x' = \mathcal{T}(\tau)(x) $$

    For tau-independent transforms, $\tau$ is ignored. For composite transforms
    (e.g., ``Composed``), the component transforms are applied sequentially.

    Parameters
    ----------
    op : coordinax.transforms.AbstractTransform
        The transform to apply. This can be any transform type:

        - ``Translate``: Spatial translation (point geometry)
        - ``Rotate``: Spatial rotation
        - ``Identity``: No-op
        - ``Composed``: Sequential composition

    tau : Any
        Parameter for tau-dependent transforms. Pass ``None`` for
        tau-independent transforms.

    x : Any
        The input to transform. Supported types depend on the transform:

        - ``Array``/``ArrayLike``: Interpreted as Cartesian point data
        - ``Quantity``: Unitful array, treated as Cartesian point
        - ``Vector``: Role-aware transformation with chart preservation
        - ``CDict``: Low-level component dict

    *args, **kwargs : Any
        Additional positional/keyword arguments passed to concrete dispatches,
        e.g. ``chart``, ``rep``, ``usys``.

    Returns
    -------
    Any
        The transformed input, same type as ``x``.

    Raises
    ------
    NotImplementedError
        If no dispatch is registered for the given ``(transform, input)`` types.

    Notes
    -----
    - **Transform.__call__**: The ``__call__`` method of transforms delegates
      to this function: ``op(tau, x)`` is equivalent to ``act(op, tau, x)``.

    - **Chart inference**: When no chart is provided and the input is an
      Array or Quantity, the chart is inferred via ``coordinax.charts.guess_chart``.

    - **Composite transforms**: For ``Composed``, the component transforms are
      applied in sequence (left-to-right).

    See Also
    --------
    coordinax.transforms.act : Concrete dispatch entrypoint used in practice
    coordinax.frames.compose : Compose two transforms into one
    coordinax.frames.simplify : Simplify a transform to canonical form

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    **Apply a rotation to a Quantity:**

    >>> op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> q = u.Q([1, 0, 0], "km")
    >>> cxfm.act(op, None, q).round(3)
    Q([0., 1., 0.], 'km')

    **Apply a translation to a Quantity (usys required):**

    >>> import jax.numpy as jnp
    >>> op = cxfm.Translate.from_([1, 0, 0], "km")
    >>> x = jnp.asarray([1.0, 0.0, 0.0])  # metres (dimensionless array)
    >>> cxfm.act(op, None, x, usys=u.unitsystems.si).round(3)
    Array([1001.,    0.,    0.], dtype=float64)

    **Composite transform:**

    >>> R = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> T = cxfm.Translate.from_([1, 0, 0], "km")
    >>> op = R | T  # rotate then translate
    >>> cxfm.act(op, None, q).round(3)
    Q([1., 1., 0.], 'km')

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def pushforward(*args: Any, **kwargs: Any) -> Any:
    r"""Apply the frozen-$\tau$ spatial differential of a transform.

    For a transform with point action $\phi(\tau, x)$, this computes the
    pushforward of a tangent vector $v$ anchored at the base point ``at``:

    $$ v' = \partial_x \phi(\tau, \cdot)\big|_{\mathrm{at}} \cdot v $$

    holding the time parameter fixed. This is the transformation law for
    `Displacement` data (a same-$\tau$ point difference) and coincides with
    `act` on all tangent kinds for time-independent transforms.

    Contrast with `act` on kinematic tangent data (velocity, acceleration,
    ...), which is the full *prolongation* and includes $\partial_\tau \phi$
    terms for time-dependent transforms.

    Canonical signature::

        pushforward(op, tau, v, chart, rep, /, *, at=None, usys=None)

    ``at`` is required whenever the transform's point action is nonlinear in
    the chart coordinates (e.g. any transform in a non-Cartesian chart);
    linear/affine fast paths may not need it.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.transforms as cxfm

    A displacement is invariant under any translation, even a time-dependent
    one (the Jacobian of a translation is the identity):

    >>> delta = lambda t: {"x": u.Q(3.0, "km/s") * t, "y": u.Q(0.0, "km"),
    ...                    "z": u.Q(0.0, "km")}
    >>> op = cxfm.Translate(delta, chart=cxc.cart3d)
    >>> d = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(0.0, "km")}
    >>> at = {"x": u.Q(0.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")}
    >>> cxfm.pushforward(op, u.Q(5.0, "s"), d, cxc.cart3d, cxr.coord_disp, at=at)
    {'x': Q(1., 'km'), 'y': Q(2., 'km'), 'z': Q(0., 'km')}

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def prolong(*args: Any, **kwargs: Any) -> Any:
    r"""Apply the kinematic (jet) prolongation of a transform to a jet.

    A *jet* is a dictionary of curve data keyed by time-derivative order:
    ``{0: q, 1: v, 2: a, ...}`` — slot 0 is the base point (position), slot 1
    the velocity, slot 2 the acceleration. All-integer keys keep the jet a
    valid JAX pytree. Displacement data (a same-$\tau$ point difference, not
    a curve derivative) is never a jet slot; it transforms by `pushforward`.

    For a transform with point action $\phi(\tau, x)$ acting on a curve
    $x(\tau)$ via $x'(\tau) = \phi(\tau, x(\tau))$, the transformed jet slots
    are the total $\tau$-derivatives:

    $$ v' = \partial_\tau \phi + \partial_x \phi \cdot v, \qquad
       a' = \partial_{\tau\tau}\phi + 2\,\partial_\tau\partial_x\phi \cdot v
            + \partial_{xx}\phi(v, v) + \partial_x \phi \cdot a, \ \ldots $$

    This is the joint, order-consistent application of `act` to a full
    phase-space state — the natural verb for `coordinax.Coordinate` bundles
    and for time-dependent transforms, where higher slots depend on all
    lower ones.

    Canonical signature::

        prolong(op, tau, jet, chart, /, *, usys=None) -> jet

    See Also
    --------
    act : per-slot application (delegates here for time-dependent transforms)
    pushforward : the frozen-tau spatial differential

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def compose(*args: Any, **kwargs: Any) -> Any:
    """Compose two frame transforms into a single transform.

    Examples
    --------
    >>> import coordinax.transforms as cxfm
    >>> import unxt as u

    >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
    >>> rotate = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))

    >>> cxfm.compose(shift, rotate)
    Composed(( Translate(...), Rotate(...) ))

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def simplify(*args: Any, **kwargs: Any) -> Any:
    """Simplify a transform to a canonical form.

    This function takes a transform and attempts to simplify it, returning a
    new, potentially simpler transform. For example, a ``Translate`` with zero
    delta simplifies to ``Identity``.

    Notes
    -----
    In general this cannot be called in a JIT'ed context because it generally
    requires inspecting values to determine if simplifications are possible.

    This function uses multiple dispatch. Each operator type registers its own
    simplification rules.

    To see all available dispatches::

        >>> import coordinax.transforms as cxfm
        >>> cxfm.simplify.methods  # doctest: +SKIP
        List of 7 method(s):
            [0] simplify(...)

    Examples
    --------
    >>> import coordinax.transforms as cxfm

    **Identity (already simple):**

    >>> op = cxfm.Identity()
    >>> cxfm.simplify(op) is op
    True

    **Translate with zero delta:**

    >>> op = cxfm.Translate.from_([0, 0, 0], "m")
    >>> cxfm.simplify(op)
    Identity()

    **Translate with non-zero delta (no simplification):**

    >>> op = cxfm.Translate.from_([1, 2, 3], "m")
    >>> simplified = cxfm.simplify(op)
    >>> type(simplified).__name__
    'Translate'

    **Rotate with identity matrix:**

    >>> import unxt as u
    >>> op = cxfm.Rotate.from_euler("z", u.Q(0, "deg"))
    >>> cxfm.simplify(op)
    Identity()

    """
    raise NotImplementedError  # pragma: no cover
