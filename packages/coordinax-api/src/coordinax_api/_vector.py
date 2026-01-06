"""Vector API for coordinax.

Copyright (c) 2023 Coordinax Devs. All rights reserved.
"""

__all__ = ("vconvert",)

from typing import Any

import plum


@plum.dispatch.abstract
def vconvert(target: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Transform the current vector to the target representation.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.vecs as cxv
    >>> import coordinax as cx

    ## 1D:

    - Array-valued:

    >>> params = {"x": jnp.array([1.0, 2.0])}
    >>> cx.vconvert(cx.r.Radial1D, cx.r.CartesianPos1D, params)
    ({'r': Array([1., 2.], dtype=float32)}, {})

    - Quantity-valued:

    >>> params = {"x": u.Q([1.0, 2.0], "m")}
    >>> cx.vconvert(cx.r.Radial1D, cx.r.CartesianPos1D, params)
    ({'r': Quantity(Array([1., 2.], dtype=float32), unit='m')},
     {})

    - Vector-valued:

    >>> x = cxv.Vector.from_(1, "km")
    >>> y = cxv.vconvert(cxv.Radial1D, x)
    >>> print(y)
    <Radial1D: (r) [km]
        [1]>

    ## 2D:

    - Array-valued:

    Without unit information "phi" is assumed to be in radians.

    >>> params = {"r": jnp.array([1.0, 2.0]), "phi": jnp.array(3)}
    >>> cx.vconvert(cx.r.Cart2D, cx.r.Polar2D, params)
    ({'x': Array([-0.9899925, -1.979985 ], dtype=float32),
      'y': Array([0.14112, 0.28224], dtype=float32)},
     {})

    We can provide that unit information so that "phi" is in degrees:

    >>> usys = u.unitsystem("kpc", "deg")
    >>> cx.vconvert(cx.r.Cart2D, cx.r.Polar2D, params, units=usys)
    ({'x': Array([0.9986295, 1.997259 ], dtype=float32),
      'y': Array([0.05233596, 0.10467192], dtype=float32)},
     {})

    - Quantity-valued:

    >>> params = {"r": u.Q([1.0, 2.0], "m"), "phi": u.Q(3, "deg")}
    >>> cx.vconvert(cx.r.Cart2D, cx.r.Polar2D, params)
    ({'x': Quantity(Array([0.9986295, 1.997259 ], dtype=float32), unit='m'),
      'y': Quantity(Array([0.05233596, 0.10467192], dtype=float32), unit='m')},
     {})

    - Vector-valued:

    >>> x = cx.Vector.from_([3, 4], "km")
    >>> y = cxv.vconvert(cxv.polar2d, x)
    >>> print(y)
    <Polar2D: (r[km], phi[rad])
        [5.    0.927]>

    >>> y = cxv.vconvert(cxv.Polar2D, x, units=u.unitsystem("m", "deg"))
    >>> print(y)
    <Polar2D: (r[m], phi[deg])
        [5000.     53.13]>

    ## 3D:

    - Array-valued:

    >>> params = {"x": jnp.array([1.0, 2.0]), "y": jnp.array([3.0, 4.0]),
    ...           "z": jnp.array([5.0, 6.0])}
    >>> params, aux = cxv.vconvert(cxv.Spherical3D, cxv.Cart3D, params)
    >>> jax.tree.map(lambda x: jnp.round(x, 4), params)
    {'phi': Array([1.249 , 1.1071], dtype=float32),
     'r': Array([5.9161   , 7.4832997], dtype=float32),
     'theta': Array([0.5639, 0.6405], dtype=float32)}

    - Quantity-valued:

    >>> params = {"x": u.Q([1.0, 2.0], "m"), "y": u.Q([3.0, 4.0], "m"),
    ...           "z": u.Q([5.0, 6.0], "m")}
    >>> params, aux = cxv.vconvert(cxv.Spherical3D, cxv.Cart3D, params)
    >>> jax.tree.map(lambda x: jnp.round(x, 4), params)
    {'phi': Quantity(Array([1.249 , 1.1071], dtype=float32), unit='rad'),
     'r': Quantity(Array([5.9161   , 7.4832997], dtype=float32), unit='m'),
     'theta': Quantity(Array([0.5639, 0.6405], dtype=float32), unit='rad')}

    - Vector-valued:

    >>> x = cxv.Cart3D.from_([[1, 3, 5], [2, 4, 6]], "km")
    >>> y = cxv.vconvert(cxv.Spherical3D, x)
    >>> print(y)
    <Spherical3D: (r[km], theta[rad], phi[rad])
        [[5.916 0.564 1.249]
         [7.483 0.641 1.107]]>

    """
    raise NotImplementedError  # pragma: no cover
