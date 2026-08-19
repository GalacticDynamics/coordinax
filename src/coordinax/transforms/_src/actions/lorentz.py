"""Lorentz boost operator for Minkowski spacetime."""

__all__ = ("LorentzBoost",)

from dataclasses import replace

from jaxtyping import Array, Shaped
from typing import Any, final

import equinox as eqx
import plum
from astropy.units import UnitConversionError

import quaxed.numpy as jnp
import unxt as u

from .base import AbstractTransform
from .identity import identity
from .linear import AbstractLinearTransform
from coordinax.transforms._src import groups

#: Speed of light, used only to convert a velocity into a dimensionless beta.
#: `MinkowskiCT` measures time as ``ct`` in length units, so the chart-native
#: boost parameter is beta itself and no constant is needed on the main path.
_C = u.Q(299792458.0, "m/s")

_MSG_SUPERLUMINAL = (
    "LorentzBoost requires a subluminal boost: |beta| must be < 1 "
    "(equivalently |v| < c)."
)


_MSG_ZERO_DIRECTION = (
    "LorentzBoost.from_rapidity requires a non-zero `direction`; the zero "
    "vector has no boost axis to normalise onto."
)


def _as_beta(b: Any, /) -> Shaped[Array, "3"]:
    """Normalise ``beta`` to a bare array, requiring it to be dimensionless.

    ``jnp`` here is `quaxed.numpy`, whose `asarray` is unit-aware and returns a
    `~unxt.Quantity` unchanged. A plain ``jnp.asarray`` converter therefore did
    nothing for a `~unxt.Quantity`, storing one in a field annotated `Array`;
    the failure surfaced much later and unrecognisably, inside `matrix`.

    A dimensionless quantity is stripped. A velocity is refused rather than
    reinterpreted: ``0.6 m/s`` is not ``0.6 c``, and taking its number would be
    wrong by eight orders of magnitude while looking perfectly ordinary.
    """
    if isinstance(b, u.AbstractQuantity):
        try:
            b = u.ustrip("", b)
        except UnitConversionError as e:
            msg = (
                f"LorentzBoost `beta` is a velocity in units of c, and so "
                f"dimensionless; got {u.unit_of(b)}. For a velocity use "
                f"`LorentzBoost.from_velocity(v)`, which divides by c."
            )
            raise ValueError(msg) from e
    return jnp.asarray(b, dtype=float)


@final
class LorentzBoost(AbstractLinearTransform):
    r"""Operator for Lorentz boosts of Minkowski spacetime.

    A boost is the linear isometry of $\mathbb{R}^{1,3}$ relating two inertial
    frames in relative motion. In the canonical
    {class}`~coordinax.charts.MinkowskiCT` chart $(ct, x, y, z)$ it acts as
    $X \mapsto \Lambda X$ with

    $$
    \Lambda(\boldsymbol\beta) =
    \begin{pmatrix}
      \gamma & \gamma\boldsymbol\beta^\top \\
      \gamma\boldsymbol\beta &
        I + (\gamma - 1)\hat{\boldsymbol\beta}\hat{\boldsymbol\beta}^\top
    \end{pmatrix},
    \qquad \gamma = \frac{1}{\sqrt{1 - \beta^2}},
    $$

    which satisfies $\Lambda^\top \eta \Lambda = \eta$ for the Minkowski metric
    $\eta = \mathrm{diag}(-1, 1, 1, 1)$ — that invariance *is* the defining
    property, and it is asserted in the test suite rather than merely claimed.

    **Sign convention.** This is the *active* convention, matching
    {class}`~coordinax.transforms.Boost` (the Galilean boost): a boost with
    parameter $\boldsymbol\beta$ carries an event to where it would be for an
    object given velocity $+\boldsymbol\beta c$.

    **Relation to** {class}`~coordinax.transforms.Boost`. The two are *not* the
    same map in a limit, and it is worth being precise about how they differ.
    `Boost` acts on 3-space with $\tau$ as an external parameter,
    $x \mapsto x + \Delta v\,\tau$, leaving $\tau$ alone. `LorentzBoost` acts on
    the 4-space itself, and mixes $ct$ into the spatial components and back:

    $$ ct' = \gamma(ct + \boldsymbol\beta\cdot\mathbf{x}). $$

    At small $\beta$ the *spatial* action does reduce to `Boost`'s, but the
    temporal mixing $\boldsymbol\beta\cdot\mathbf{x}$ is **first order** in
    $\beta$, not a higher-order correction — it is the relativity of
    simultaneity, exactly the part the Galilean group lacks. The genuine
    Galilean limit is $c \to \infty$ at fixed $v$, where that term vanishes
    *relative to* $ct$.

    **Time dependence.** `LorentzBoost.is_time_dependent` is `False`, while
    {attr}`Boost.is_time_dependent` is `True`. That is not an oversight: for the
    Galilean boost, time is a parameter *outside* the manifold and the point
    action genuinely varies with it, whereas here $ct$ is a *coordinate of the
    manifold* and $\Lambda$ is a constant matrix. A boost whose rapidity varies
    with $\tau$ — an accelerating frame — is spelled the same way as any other
    time-dependent transform, by wrapping a builder:

    >>> import equinox as eqx
    >>> import quaxed.numpy as jnp
    >>> import coordinax.transforms as cxfm

    >>> class UniformlyAccelerating(eqx.Module):
    ...     rate: jnp.ndarray
    ...     def __call__(self, tau):
    ...         return cxfm.LorentzBoost(self.rate * tau)

    >>> op = cxfm.TimeDep(UniformlyAccelerating(jnp.asarray([0.1, 0.0, 0.0])))
    >>> op.is_time_dependent
    True

    Because ``beta`` is an ordinary pytree leaf rather than a callable, ``rate``
    stays differentiable and vmappable through that builder -- which is exactly
    what the `TimeDep` refactor exists to make possible.

    **Parameterisation.** Because ``MinkowskiCT`` measures time as $ct$ in
    *length* units, the chart-native boost parameter is the dimensionless
    $\boldsymbol\beta = \mathbf{v}/c$; the matrix is then dimensionless too and
    needs no speed of light. Use {meth}`from_velocity` to build one from a
    velocity in ordinary units, or {meth}`from_rapidity` for the additive
    parameterisation.

    Parameters
    ----------
    beta : Array[float, (3,)]
        The boost velocity in units of $c$. Must satisfy ``|beta| < 1``.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.transforms as cxfm

    A boost of $\beta = 0.6$ along $x$ has $\gamma = 1.25$:

    >>> op = cxfm.LorentzBoost([0.6, 0.0, 0.0])
    >>> float(op.gamma.round(3))
    1.25

    Applied to an event on the light cone, it stays on the light cone:

    >>> ev = {"ct": u.Q(1.0, "m"), "x": u.Q(1.0, "m"),
    ...       "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> out = cxfm.act(op, None, ev, cxc.minkowskict, cxr.point)
    >>> float(out["ct"].ustrip("m").round(3)), float(out["x"].ustrip("m").round(3))
    (2.0, 2.0)

    The inverse is the boost with the opposite velocity:

    >>> op.inverse.beta.round(3)
    Array([-0.6, -0. , -0. ], dtype=float64)

    """

    beta: Shaped[Array, "3"] = eqx.field(converter=_as_beta)
    """Boost velocity in units of ``c`` (dimensionless 3-vector)."""

    @classmethod
    def groups(cls) -> frozenset[type]:
        """Return the groups to which this map belongs."""
        del cls
        return frozenset(
            (groups.ProperOrthochronousLorentzGroup, groups.DiffeomorphismGroup)
        )

    # -----------------------------------------------------
    # Constructors

    @classmethod
    def from_velocity(cls, v: u.AbstractQuantity, /) -> "LorentzBoost":
        """Build a boost from a velocity 3-vector in ordinary units.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.transforms as cxfm

        Half the speed of light along ``x``:

        >>> op = cxfm.LorentzBoost.from_velocity(
        ...     u.Q([149896229.0, 0.0, 0.0], "m/s")
        ... )
        >>> op.beta.round(3)
        Array([0.5, 0. , 0. ], dtype=float64)

        """
        return cls(u.ustrip("", v / _C))

    @classmethod
    def from_rapidity(
        cls, rapidity: Any, direction: Any = (1.0, 0.0, 0.0), /
    ) -> "LorentzBoost":
        r"""Build a boost from a rapidity $\phi$ along ``direction``.

        Rapidity is the additive boost parameter: $\beta = \tanh\phi$, and
        collinear boosts compose by adding rapidities. ``direction`` need not be
        normalised.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import coordinax.transforms as cxfm

        >>> op = cxfm.LorentzBoost.from_rapidity(jnp.arctanh(0.6))
        >>> op.beta.round(3)
        Array([0.6, 0. , 0. ], dtype=float64)

        Rapidities add, so composing two boosts of ``phi`` gives ``2 phi``:

        >>> phi = 0.3
        >>> single = cxfm.LorentzBoost.from_rapidity(phi)
        >>> double = cxfm.LorentzBoost.from_rapidity(2 * phi)
        >>> bool(jnp.allclose(double.rapidity, 2 * single.rapidity))
        True

        """
        d = jnp.asarray(direction, dtype=float)
        norm = jnp.linalg.norm(d)
        # A zero direction has no boost axis to normalise onto; dividing would
        # give `nan` betas that then propagate silently into every matrix entry.
        norm = eqx.error_if(norm, norm == 0.0, _MSG_ZERO_DIRECTION)
        return cls(jnp.tanh(jnp.asarray(rapidity, dtype=float)) * (d / norm))

    # -----------------------------------------------------
    # Derived quantities

    @property
    def speed(self) -> Array:
        r"""The boost speed $|\boldsymbol\beta|$, in units of ``c``.

        Examples
        --------
        >>> import coordinax.transforms as cxfm
        >>> float(cxfm.LorentzBoost([0.6, 0.0, 0.0]).speed.round(3))
        0.6

        """
        return jnp.linalg.norm(self.beta)

    @property
    def gamma(self) -> Array:
        r"""The Lorentz factor $\gamma = 1/\sqrt{1 - \beta^2}$.

        Examples
        --------
        >>> import coordinax.transforms as cxfm
        >>> float(cxfm.LorentzBoost([0.6, 0.0, 0.0]).gamma.round(3))
        1.25

        """
        beta_sq = jnp.sum(self.beta**2)
        beta_sq = eqx.error_if(beta_sq, beta_sq >= 1.0, _MSG_SUPERLUMINAL)
        return 1.0 / jnp.sqrt(1.0 - beta_sq)

    @property
    def rapidity(self) -> Array:
        r"""The rapidity $\phi = \mathrm{arctanh}\,|\boldsymbol\beta|$.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import coordinax.transforms as cxfm
        >>> op = cxfm.LorentzBoost([0.6, 0.0, 0.0])
        >>> bool(jnp.allclose(op.rapidity, jnp.arctanh(0.6)))
        True

        """
        # `arctanh` returns inf at |beta| == 1 and nan beyond it. Guard on the
        # same condition as `gamma`, so every derived quantity reports the same
        # superluminal error rather than one of them leaking a non-finite value.
        speed = self.speed
        speed = eqx.error_if(speed, speed >= 1.0, _MSG_SUPERLUMINAL)
        return jnp.arctanh(speed)

    # -----------------------------------------------------

    @property
    def inverse(self) -> "LorentzBoost":
        """The inverse boost, with the opposite velocity.

        Examples
        --------
        >>> import coordinax.transforms as cxfm
        >>> op = cxfm.LorentzBoost([0.6, 0.0, 0.0])
        >>> op.inverse.beta.round(3)
        Array([-0.6, -0. , -0. ], dtype=float64)

        """
        return replace(self, beta=-self.beta)

    def __neg__(self) -> "LorentzBoost":
        """Negate the boost velocity (same as `inverse`).

        Examples
        --------
        >>> import coordinax.transforms as cxfm
        >>> (-cxfm.LorentzBoost([0.6, 0.0, 0.0])).beta.round(3)
        Array([-0.6, -0. , -0. ], dtype=float64)

        """
        return self.inverse

    # -----------------------------------------------------

    @property
    def _raw_matrix(self) -> Array:
        r"""The $4\times4$ boost matrix $\Lambda(\boldsymbol\beta)$.

        Examples
        --------
        >>> import coordinax.transforms as cxfm
        >>> cxfm.LorentzBoost([0.6, 0.0, 0.0])._raw_matrix.round(2)
        Array([[1.25, 0.75, 0.  , 0.  ],
               [0.75, 1.25, 0.  , 0.  ],
               [0.  , 0.  , 1.  , 0.  ],
               [0.  , 0.  , 0.  , 1.  ]], dtype=float64)

        """
        beta = self.beta
        gamma = self.gamma
        beta_sq = jnp.sum(beta**2)

        # The spatial block is I + (gamma - 1) * outer(beta, beta) / beta^2.
        # At beta == 0 that ratio is 0/0; guard the denominator and select the
        # identity there, so a zero boost is the identity rather than nan.
        safe_beta_sq = jnp.where(beta_sq > 0.0, beta_sq, 1.0)
        spatial = jnp.eye(3) + (gamma - 1.0) * jnp.outer(beta, beta) / safe_beta_sq

        top_row = jnp.concatenate([gamma[None], gamma * beta])
        bottom = jnp.concatenate([(gamma * beta)[:, None], spatial], axis=1)
        return jnp.concatenate([top_row[None, :], bottom], axis=0)


@plum.dispatch
def simplify(
    op: LorentzBoost, /, *, approx: bool = True, **kw: Any
) -> AbstractTransform:
    """Simplify a Lorentz boost to identity when its velocity is zero.

    Every other transform has had this rule; `LorentzBoost` did not, and
    `simplify` dispatches per operator with no generic fallback -- so
    ``simplify`` of a boost, or of any `~coordinax.transforms.Composed`
    containing one, raised `NotFoundLookupError` rather than returning the
    operator unchanged.

    The zero-velocity check inspects values, so it is skipped when
    ``approx=False``; the point of the rule is that the ``approx=False`` path
    now returns ``op`` instead of raising.

    Examples
    --------
    >>> import coordinax.transforms as cxfm

    A boost with velocity is left alone:

    >>> cxfm.simplify(cxfm.LorentzBoost([0.6, 0.0, 0.0]))
    LorentzBoost(...)

    A zero boost is the identity:

    >>> cxfm.simplify(cxfm.LorentzBoost([0.0, 0.0, 0.0]))
    Identity()

    """
    if approx and jnp.allclose(op.beta, jnp.zeros_like(op.beta), **kw):
        return identity
    return op
