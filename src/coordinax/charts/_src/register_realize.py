"""Register API for realization maps."""

__all__: tuple[str, ...] = ()

from collections.abc import Callable
from typing import Any

import plum

import unxt as u

import coordinax.api.charts as api
from .base import AbstractChart
from .custom_types import CDict
from coordinax.internal.custom_types import OptUSys

_DIMLESS: u.AbstractUnit = u.unit("")


@plum.dispatch
def point_realization_map(
    to_chart: AbstractChart,  # type: ignore[type-arg]
    from_chart: AbstractChart,  # type: ignore[type-arg]
    p: Any,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    """Map point coordinates from one chart to another using the transition map.

    The default and conservative assumption is that the mapping is being done
    on the same atlas (in the same manifold).

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    2D polar to Cartesian:

    >>> p_polar = {"r": u.Q(2.0, "m"), "theta": u.Angle(jnp.pi / 4, "rad")}
    >>> cxc.point_realization_map(cxc.cart2d, cxc.polar2d, p_polar)
    {'x': Q(1.41421356, 'm'), 'y': Q(1.41421356, 'm')}

    3D spherical to Cartesian:

    >>> p_sph = {"theta": u.Angle(jnp.pi / 2, "rad"),
    ...          "phi": u.Angle(0.0, "rad"), "r": u.Q(5.0, "km")}
    >>> cxc.point_realization_map(cxc.cart3d, cxc.sph3d, p_sph)
    {'x': Q(5., 'km'), 'y': Q(0., 'km'), 'z': Q(3.061617e-16, 'km')}

    Cartesian to cylindrical:

    >>> p_xyz = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(5.0, "m")}
    >>> cxc.point_realization_map(cxc.cyl3d, cxc.cart3d, p_xyz)
    {'rho': Q(5., 'm'), 'phi': Q(0.92729522, 'rad'), 'z': Q(5., 'm')}

    Identity mapping keeps components unchanged:

    >>> cxc.point_realization_map(cxc.cart3d, cxc.cart3d, p_xyz)
    {'x': Q(3., 'm'), 'y': Q(4., 'm'), 'z': Q(5., 'm')}

    """
    return api.point_transition_map(to_chart, from_chart, p, usys=usys)


@plum.dispatch
def point_realization_map(
    to_chart: AbstractChart,  # type: ignore[type-arg]
    from_chart: AbstractChart,  # type: ignore[type-arg]
    /,
    **fixed_kwargs: Any,
) -> Callable[..., Any]:
    """Return a partial function for point transformation.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Coordinates without units are the default.

    >>> p = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> map = cxc.point_realization_map(cxc.sph3d, cxc.cart3d)
    >>> map(p)
    {'r': Q(1., 'm'), 'theta': Q(1.57079633, 'rad'), 'phi': Q(0., 'rad')}

    Coordinates without units are also accepted, interpreted having units of the
    `unxt.AbstractUnitSystem`, which must be passed.

    >>> p = {"x": 1.0, "y": 0.0, "z": 0.0}
    >>> map = cxc.point_realization_map(cxc.sph3d, cxc.cart3d,
    ...                                usys=u.unitsystems.si)
    >>> map(p)
    {'r': Array(1., dtype=float64, ...),
     'theta': Array(1.57079633, dtype=float64),
     'phi': Array(0., dtype=float64, ...)}

    `unxt.Quantity` inputs are also accepted, and are interpreted as being in
    Cartesian coordinates.

    >>> p = u.Q([1.0, 0.0, 0.0], "m")
    >>> map = cxc.point_realization_map(cxc.sph3d, cxc.cart3d)
    >>> map(p)
    QuantityMatrix([1.        , 1.57079633, 0.        ], '(m, rad, rad)')

    Array-Like inputs are interpreted as Cartesian coordinates with units from
    the required `unxt.AbstractUnitSystem`.

    >>> p = [1.0, 0.0, 0.0]
    >>> map = cxc.point_realization_map(cxc.sph3d, cxc.cart3d, usys=u.unitsystems.si)
    >>> map(p)
    Array([1.        , 1.57079633, 0.        ], dtype=float64)

    """
    # NOTE: lambda is much faster than ft.partial here
    return lambda *args, **kw: api.point_realization_map(
        to_chart, from_chart, *args, **fixed_kwargs, **kw
    )


# ---------------------------------------------------------------------------


@plum.dispatch
def realize_cartesian(
    chart: AbstractChart,  # type: ignore[type-arg]
    data: Any,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    r"""Realize a point in canonical ambient Cartesian coordinates.

    This method evaluates the chart's (optional) **ambient realization map**

    $$ X: V \subset \mathbb{R}^n \to \mathbb{R}^m $$

    mapping point-role coordinates in this chart to point-role coordinates
    in the chart's distinguished ambient Cartesian chart,
    ``chart.cartesian``.

    - This is **point-role only**..
    - For parameter-free Euclidean reparameterizations (e.g. spherical or
      cylindrical charts on $\mathbb{R}^3$), this is typically a canonical map.
    - For charts whose realization depends on additional geometric data
      (e.g. a 2-sphere in $\mathbb{R}^3$ requiring a radius), charts may not
      provide a canonical realization; in such cases the underlying
      transition rule to ``self.cartesian`` may be unregistered and this
      method will fail. In that case, users should use
      `coordinax.manifolds.AbstractManifold.realize_cartesian` with an
      appropriate embedded manifold chart or construct a
      `coordinax.embeddings.EmbeddedChart` which provides a custom
      realization map.

    Parameters
    ----------
    chart:
        The chart on which to evaluate the ambient realization map.
    data:
        Point coordinates in this chart, represented as a ``CDict`` mapping
        component names to arrays or quantities.
    usys:
        Optional unit system used to interpret unitful inputs.

    Returns
    -------
    CDict
        Point coordinates in the canonical ambient Cartesian chart.

    Raises
    ------
    Exception
        If no point transition rule exists from this chart to
        ``self.cartesian`` (e.g. for charts requiring embedding parameters).

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Spherical to ambient Cartesian:

    >>> at = {"r": u.Q(2.0, "m"), "theta": u.Q(1.5707963, "rad"),
    ...       "phi": u.Q(0.0, "rad")}
    >>> cxc.sph3d.realize_cartesian(at)
    {'x': Q(2., 'm'), 'y': Q(0., 'm'), 'z': Q(5...e-08, 'm')}

    Cartesian to Cartesian is the identity:

    >>> at = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"),
    ...       "z": u.Q(3.0, "m")}
    >>> cxc.cart3d.realize_cartesian(at)
    {'x': Q(1., 'm'), 'y': Q(2., 'm'), 'z': Q(3., 'm')}

    """
    return api.point_realization_map(chart.cartesian, chart, data, usys=usys)


@plum.dispatch
def unrealize_cartesian(
    chart: AbstractChart,  # type: ignore[type-arg]
    data: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> CDict:
    """Invert the ambient Cartesian realization on the chart domain.

    This method applies the inverse of the chart's ambient realization map
    (when defined) to convert point-role coordinates in ``self.cartesian``
    to point-role coordinates in this chart.

    - This is **point-role only**.
    - The inverse map may be undefined or multi-valued globally; this method
        represents the inverse only on the chart's intended domain.
    - For charts whose realization depends on additional geometric data
        (e.g. embedding parameters), the corresponding transition rule from
        ``self.cartesian`` may be unregistered and this method will fail. In
        that case, users should use
        `coordinax.manifolds.AbstractManifold.realize_cartesian` with an
        appropriate embedded manifold chart or construct a
        `coordinax.embeddings.EmbeddedChart` which provides a custom
        realization map

    Parameters
    ----------
    chart:
        The chart on which to evaluate the inverse realization map.
    data:
        Point coordinates in the canonical ambient Cartesian chart,
        represented as a ``CDict``.
    usys:
        Optional unit system used to interpret unitful inputs.

    Returns
    -------
    CDict
        Point coordinates in this chart.

    Raises
    ------
    Exception
        If no point transition rule exists from ``self.cartesian`` to this
        chart.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import unxt as u

    Cartesian point back to spherical:

    >>> cart_pt = {"x": u.Q(0.0, "m"), "y": u.Q(2.0, "m"),
    ...           "z": u.Q(0.0, "m")}
    >>> cxc.sph3d.unrealize_cartesian(cart_pt)
    {'r': Q(2., 'm'),
        'theta': Q(1.57079633, 'rad'),
        'phi': Q(1.57079633, 'rad')}

    Cartesian to Cartesian is the identity:

    >>> at = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"),
    ...       "z": u.Q(3.0, "m")}
    >>> cxc.cart3d.unrealize_cartesian(at)
    {'x': Q(1., 'm'),
        'y': Q(2., 'm'),
        'z': Q(3., 'm')}

    """
    return api.point_realization_map(chart, chart.cartesian, data, usys=usys)
