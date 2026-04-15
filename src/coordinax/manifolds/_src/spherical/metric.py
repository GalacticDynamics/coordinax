"""Two-sphere manifold."""

__all__ = ("HyperSphericalMetric",)

import dataclasses

from typing import final

import jax
import jax.numpy as jnp

import unxt as u
from unxt.quantity import AllowValue

import coordinax.charts as cxc
from coordinax.internal import QuantityMatrix, UnitsMatrix
from coordinax.internal.custom_types import CDict, OptUSys
from coordinax.manifolds._src.base import AbstractMetric


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class HyperSphericalMetric(AbstractMetric):
    r"""Round metric on the unit $n$-sphere $S^{n-1}$ in standard spherical coordinates.

    The round metric on $S^2$ in the $(\theta, \phi)$ spherical chart is

    $$g = \begin{pmatrix} 1 & 0 \\ 0 & \sin^2\theta \end{pmatrix}.$$

    Parameters
    ----------
    ndim : int
        Intrinsic dimension of the sphere (e.g. ``ndim=2`` for $S^2$).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> m = cxm.HyperSphericalMetric(2)
    >>> at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0.0)}
    >>> m.metric_matrix(cxc.sph2, at=at)
    Array([[1., 0.],
           [0., 1.]], dtype=float64)

    The signature is ``(1,) * ndim`` for this positive-definite metric:

    >>> m.signature
    (1, 1)
    >>> m.ndim
    2

    """

    ndim: int
    """Intrinsic dimension of the sphere."""

    @property
    def signature(self) -> tuple[int, ...]:
        """Metric signature: ``(1,) * ndim`` — the round sphere metric is Riemannian."""
        return (1,) * self.ndim

    def metric_matrix(
        self, chart: cxc.AbstractChart, /, *, at: CDict, usys: OptUSys = None
    ) -> QuantityMatrix | jax.Array:
        r"""Metric matrix $g = \operatorname{diag}(g_0, \ldots, g_{n-1})$ at ``at``.

        The diagonal entries follow the cumulative-sine rule:

        $$g_{kk} = \prod_{j=0}^{k-1} \sin^2(\theta_j)$$

        so $g_{00} = 1$, $g_{11} = \sin^2\theta_0$,
        $g_{22} = \sin^2\theta_0\,\sin^2\theta_1$, etc.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.charts as cxc
        >>> import coordinax.manifolds as cxm

        Bare arrays (angles in radians) → plain ``jax.Array``:

        >>> M = cxm.HyperSphericalMetric(2)
        >>> at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0.0)}
        >>> M.metric_matrix(cxc.sph2, at=at)
        Array([[1., 0.],
               [0., 1.]], dtype=float64)

        {class}`~unxt.Quantity` angles (radians) →
        {class}`~coordinax.internal.QuantityMatrix`:

        >>> at = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
        >>> M.metric_matrix(cxc.sph2, at=at)
        QuantityMatrix([[1., 0.],
                        [0., 1.]], '((, ), (, ))')

        {class}`~unxt.Quantity` angles in degrees are converted automatically:

        >>> at = {"theta": u.Angle(90.0, "deg"), "phi": u.Angle(0.0, "deg")}
        >>> M.metric_matrix(cxc.sph2, at=at)
        QuantityMatrix([[1., 0.],
                        [0., 1.]], '((, ), (, ))')

        """
        components = chart.components
        is_qty = any(map(u.quantity.is_any_quantity, at.values()))
        ang_unit = usys["angle"] if usys is not None else u.unit("rad")

        # Seed with a scalar of the right dtype/shape from the first component.
        dtype = jnp.promote_types(*(v.dtype for v in at.values()))
        cumprod = jnp.ones((), dtype=dtype)
        diag_entries = [cumprod]
        for k in components[:-1]:
            ang = u.ustrip(AllowValue, u.uconvert_value(u.unit("rad"), ang_unit, at[k]))
            cumprod = cumprod * jnp.sin(ang) ** 2
            diag_entries.append(cumprod)
        G_arr = jnp.diag(jnp.stack(diag_entries))

        if not is_qty:
            return G_arr

        n = self.ndim
        unit_tup = tuple(tuple(u.unit("") for _ in range(n)) for _ in range(n))
        return QuantityMatrix(G_arr, unit=UnitsMatrix(unit_tup))
