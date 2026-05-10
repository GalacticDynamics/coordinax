"""Minkowski spacetime manifold."""

__all__ = ("MinkowskiMetric",)

import dataclasses

from typing import final

import jax
import jax.numpy as jnp

import quaxed.numpy as qnp
import unxt as u

import coordinax.charts as cxc
from coordinax._src.base_metric import AbstractDiagonalMetric
from coordinax._src.custom_types import CDict, OptUSys
from coordinax.internal import QuantityMatrix, UnitsMatrix


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class MinkowskiMetric(AbstractDiagonalMetric):
    r"""Pseudo-Riemannian (Lorentzian) metric on Minkowski spacetime.

    In the canonical {class}`~coordinax.charts.SpaceTimeCT` chart with
    Cartesian spatial part $(ct, x, y, z)$, the Minkowski metric is

    $$\eta = \operatorname{diag}(-1, 1, 1, 1),$$

    where the time coordinate $ct = c\,t$ absorbs the speed of light so that
    all four components carry the same unit (length). The line element is

    $$ds^2 = -(d(ct))^2 + dx^2 + dy^2 + dz^2.$$

    For a general {class}`~coordinax.charts.SpaceTimeCT` chart with spatial
    sub-chart $C_s$ (e.g. spherical or cylindrical), the metric is the
    pullback

    $$g_{ij} = J^k{}_i\,\eta_{kl}\,J^l{}_j = (J^T \eta J)_{ij},$$

    where $J_{kj} = \partial(ct, x, y, z)^k / \partial q^j$ is the Jacobian
    from the given chart to the canonical Cartesian spacetime chart computed
    by :func:`~coordinax.charts.jac_pt_map`.

    **Signature.** The metric is **pseudo-Riemannian** with Lorentzian
    signature $(-1, 1, 1, 1)$ meaning one negative and three positive
    eigenvalues (convention: "mostly plus").

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    Canonical Cartesian spacetime chart:

    >>> m = cxm.MinkowskiMetric()
    >>> at = {"ct": jnp.array(0.0), "x": jnp.array(0.0),
    ...       "y": jnp.array(0.0), "z": jnp.array(0.0)}
    >>> m.metric_matrix(cxc.spacetimect, at=at).value
    Array([[-1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]], dtype=float64)

    The signature is Lorentzian (pseudo-Riemannian):

    >>> m.signature
    (-1, 1, 1, 1)

    >>> m.ndim
    4

    """

    @property
    def signature(self) -> tuple[int, ...]:
        """Metric signature: ``(-1, 1, 1, 1)`` — Lorentzian pseudo-Riemannian."""
        return (-1, 1, 1, 1)

    def metric_matrix(
        self, chart: cxc.AbstractChart, /, *, at: CDict, usys: OptUSys = None
    ) -> QuantityMatrix:
        r"""Compute the Minkowski metric tensor $g_{ij}$ at the base point ``at``.

        In the canonical Cartesian chart returns $\eta =
        \operatorname{diag}(-1, 1, 1, 1)$ directly. For other
        {class}`~coordinax.charts.SpaceTimeCT` charts computes the pullback
        $g = J^T \eta J$ via :func:`~coordinax.charts.jac_pt_map`.

        Parameters
        ----------
        chart : AbstractChart
            Coordinate chart in which to express the metric. Must support a
            ``.cartesian`` property (i.e. be a
            {class}`~coordinax.charts.SpaceTimeCT` instance).
        at : CDict
            Base point in ``chart`` coordinates at which to evaluate.
        usys : OptUSys, optional
            Unit system to use for the metric evaluation.

        Returns
        -------
        QuantityMatrix, shape (4, 4)
            Metric matrix $g_{ij}$ in the given chart basis.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import coordinax.charts as cxc
        >>> import coordinax.manifolds as cxm

        Canonical Cartesian chart — returns the flat Minkowski matrix:

        >>> m = cxm.MinkowskiMetric()
        >>> at = {"ct": jnp.array(0.0), "x": jnp.array(0.0),
        ...       "y": jnp.array(0.0), "z": jnp.array(0.0)}
        >>> m.metric_matrix(cxc.spacetimect, at=at).value
        Array([[-1.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  1.]], dtype=float64)

        Spherical spatial chart — metric becomes $\operatorname{diag}(-1, 1, r^2,
        r^2\sin^2\!\theta)$:

        >>> import unxt as u
        >>> at_sph = {"ct": u.Q(0.0, "m"), "r": u.Q(2.0, "m"),
        ...           "theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0.0, "rad")}
        >>> g = m.metric_matrix(cxc.SpaceTimeCT(cxc.sph3d), at=at_sph)
        >>> jnp.diag(g.value).round(6)
        Array([-1.,  1.,  4.,  4.], dtype=float64)

        """
        n = 4
        unit_tup = tuple(tuple(u.unit("") for _ in range(n)) for _ in range(n))

        cart_chart = chart.cartesian
        if chart == cart_chart:
            # Already the canonical Cartesian spacetime chart: η = diag(-1,1,1,1)
            return QuantityMatrix(
                jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0])),
                unit=UnitsMatrix(unit_tup),
            )

        # Pullback: g = J^T η J
        # J: jacobian of chart → cart_chart, shape (4, 4)
        J = cxc.jac_pt_map(at, chart, cart_chart, usys=usys)
        JT = qnp.transpose(J, (1, 0))
        eta = QuantityMatrix(
            jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0])),
            unit=UnitsMatrix(unit_tup),
        )
        return JT @ eta @ J  # ty: ignore[invalid-return-type]
