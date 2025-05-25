"""Copyright (c) 2023 coordinax maintainers. All rights reserved."""

__all__ = [
    "vector",
    "vconvert",
    "normalize_vector",
    "cartesian_vector_type",
    "time_derivative_vector_type",
    "time_antiderivative_vector_type",
    "time_nth_derivative_vector_type",
]

from typing import TYPE_CHECKING, Any

from plum import dispatch

if TYPE_CHECKING:
    import coordinax.vecs


@dispatch.abstract
def vector(*args: Any, **kwargs: Any) -> Any:
    """Construct a vector given the arguments."""
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def vconvert(target: type[Any], /, *args: Any, **kwargs: Any) -> Any:
    """Transform the current vector to the target vector.

    See the dispatch implementations for more details. Not all transformations
    result in the target vector type, for example
    ``vconvert(type[Cartesian3DPos], FourVector)`` will return a
    `coordinax.vecs.FourVector` with the spatial part in Cartesian coordinates.
    Likewise, `coordinax.vconvert` on `coordinax.Coordinate` instances will
    transform the contained vectors to the target type, returning a
    `coordinax.Coordinate` instance.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    ## 1D:

    - Array-valued:

    >>> params = {"x": jnp.array([1.0, 2.0])}
    >>> cxv.vconvert(cxv.RadialPos, cxv.CartesianPos1D, params)
    ({'r': Array([1., 2.], dtype=float32)}, {})

    - Quantity-valued:

    >>> params = {"x": u.Quantity([1.0, 2.0], "m")}
    >>> cxv.vconvert(cxv.RadialPos, cxv.CartesianPos1D, params)
    ({'r': Quantity(Array([1., 2.], dtype=float32), unit='m')},
     {})

    - Vector-valued:

    >>> x = cxv.CartesianPos1D.from_(1, "km")
    >>> y = cxv.vconvert(cxv.RadialPos, x)
    >>> print(y)
    <RadialPos: (r) [km]
        [1]>

    ## 2D:

    - Array-valued:

    Without unit information "phi" is assumed to be in radians.

    >>> params = {"r": jnp.array([1.0, 2.0]), "phi": jnp.array(3)}
    >>> cxv.vconvert(cxv.CartesianPos2D, cxv.PolarPos, params)
    ({'x': Array([-0.9899925, -1.979985 ], dtype=float32),
      'y': Array([0.14112, 0.28224], dtype=float32)},
     {})

    We can provide that unit information so that "phi" is in degrees:

    >>> usys = u.unitsystem("kpc", "deg")
    >>> cxv.vconvert(cxv.CartesianPos2D, cxv.PolarPos, params, units=usys)
    ({'x': Array([0.9986295, 1.997259 ], dtype=float32),
      'y': Array([0.05233596, 0.10467192], dtype=float32)},
     {})

    - Quantity-valued:

    >>> params = {"r": u.Quantity([1.0, 2.0], "m"), "phi": u.Quantity(3, "deg")}
    >>> cxv.vconvert(cxv.CartesianPos2D, cxv.PolarPos, params)
    ({'x': Quantity(Array([0.9986295, 1.997259 ], dtype=float32), unit='m'),
      'y': Quantity(Array([0.05233596, 0.10467192], dtype=float32), unit='m')},
     {})

    - Vector-valued:

    >>> x = cxv.CartesianPos2D.from_([3, 4], "km")
    >>> y = cxv.vconvert(cxv.PolarPos, x)
    >>> print(y)
    <PolarPos: (r[km], phi[rad])
        [5.    0.927]>

    >>> y = cxv.vconvert(cxv.PolarPos, x, units=u.unitsystem("m", "deg"))
    >>> print(y)
    <PolarPos: (r[m], phi[deg])
        [5000.     53.13]>

    ## 3D:

    - Array-valued:

    >>> params = {"x": jnp.array([1.0, 2.0]), "y": jnp.array([3.0, 4.0]),
    ...           "z": jnp.array([5.0, 6.0])}
    >>> params, aux = cxv.vconvert(cxv.SphericalPos, cxv.CartesianPos3D, params)
    >>> jax.tree.map(lambda x: jnp.round(x, 4), params)
    {'phi': Array([1.249 , 1.1071], dtype=float32),
     'r': Array([5.9161   , 7.4832997], dtype=float32),
     'theta': Array([0.5639, 0.6405], dtype=float32)}

    - Quantity-valued:

    >>> params = {"x": u.Quantity([1.0, 2.0], "m"),
    ...           "y": u.Quantity([3.0, 4.0], "m"),
    ...           "z": u.Quantity([5.0, 6.0], "m")}
    >>> params, aux = cxv.vconvert(cxv.SphericalPos, cxv.CartesianPos3D, params)
    >>> jax.tree.map(lambda x: jnp.round(x, 4), params)
    {'phi': Quantity(Array([1.249 , 1.1071], dtype=float32), unit='rad'),
     'r': Quantity(Array([5.9161   , 7.4832997], dtype=float32), unit='m'),
     'theta': Quantity(Array([0.5639, 0.6405], dtype=float32), unit='rad')}

    - Vector-valued:

    >>> x = cxv.CartesianPos3D.from_([[1, 3, 5], [2, 4, 6]], "km")
    >>> y = cxv.vconvert(cxv.SphericalPos, x)
    >>> print(y)
    <SphericalPos: (r[km], theta[rad], phi[rad])
        [[5.916 0.564 1.249]
         [7.483 0.641 1.107]]>

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def normalize_vector(x: Any, /) -> Any:
    """Return the unit vector."""
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def cartesian_vector_type(obj: Any, /) -> "type[coordinax.vecs.AbstractVector]":
    """Return the corresponding Cartesian vector type.

    Examples
    --------
    >>> import coordinax as cx

    >>> cx.vecs.cartesian_vector_type(cx.vecs.RadialPos)
    <class 'coordinax...CartesianPos1D'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.SphericalPos)
    <class 'coordinax...CartesianPos3D'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.RadialVel)
    <class 'coordinax...CartesianVel1D'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.TwoSphereAcc)
    <class 'coordinax...CartesianAcc2D'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.SphericalVel)
    <class 'coordinax...CartesianVel3D'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.CartesianAcc3D)
    <class 'coordinax...CartesianAcc3D'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.SphericalAcc)
    <class 'coordinax...CartesianAcc3D'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.FourVector)
    <class 'coordinax...CartesianPos3D'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.CartesianPosND)
    <class 'coordinax...CartesianPosND'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.CartesianVelND)
    <class 'coordinax...CartesianVelND'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.CartesianAccND)
    <class 'coordinax...CartesianAccND'>

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def time_derivative_vector_type(obj: Any, /) -> "type[coordinax.vecs.AbstractVector]":
    """Return the corresponding time derivative vector type.

    Examples
    --------
    >>> import coordinax as cx

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CartesianPos1D)
    <class 'coordinax...CartesianVel1D'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CartesianVel1D)
    <class 'coordinax...CartesianAcc1D'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.RadialPos)
    <class 'coordinax...RadialVel'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.RadialVel)
    <class 'coordinax...RadialAcc'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CartesianPos2D)
    <class 'coordinax...CartesianVel2D'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CartesianVel2D)
    <class 'coordinax...CartesianAcc2D'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.PolarPos)
    <class 'coordinax...PolarVel'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.PolarVel)
    <class 'coordinax...PolarAcc'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CartesianPos3D)
    <class 'coordinax...CartesianVel3D'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CartesianVel3D)
    <class 'coordinax...CartesianAcc3D'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CylindricalPos)
    <class 'coordinax...CylindricalVel'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CylindricalVel)
    <class 'coordinax...CylindricalAcc'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.SphericalPos)
    <class 'coordinax...SphericalVel'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.SphericalVel)
    <class 'coordinax...SphericalAcc'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.MathSphericalPos)
    <class 'coordinax...MathSphericalVel'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.MathSphericalVel)
    <class 'coordinax...MathSphericalAcc'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.LonLatSphericalPos)
    <class 'coordinax...LonLatSphericalVel'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.LonLatSphericalVel)
    <class 'coordinax...LonLatSphericalAcc'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.LonCosLatSphericalVel)
    <class 'coordinax...LonLatSphericalAcc'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.ProlateSpheroidalPos)
    <class 'coordinax...ProlateSpheroidalVel'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.ProlateSpheroidalVel)
    <class 'coordinax...ProlateSpheroidalAcc'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CartesianPosND)
    <class 'coordinax...CartesianVelND'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CartesianVelND)
    <class 'coordinax...CartesianAccND'>

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def time_antiderivative_vector_type(
    obj: Any, /
) -> "type[coordinax.vecs.AbstractVector]":
    """Return the corresponding time antiderivative vector type.

    Examples
    --------
    >>> import coordinax as cx

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CartesianVel1D)
    <class 'coordinax...CartesianPos1D'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CartesianAcc1D)
    <class 'coordinax...CartesianVel1D'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.RadialVel)
    <class 'coordinax...RadialPos'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.RadialAcc)
    <class 'coordinax...RadialVel'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CartesianVel2D)
    <class 'coordinax...CartesianPos2D'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CartesianAcc2D)
    <class 'coordinax...CartesianVel2D'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.PolarVel)
    <class 'coordinax...PolarPos'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.PolarAcc)
    <class 'coordinax...PolarVel'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CartesianVel3D)
    <class 'coordinax...CartesianPos3D'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CartesianAcc3D)
    <class 'coordinax...CartesianVel3D'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CylindricalVel)
    <class 'coordinax...CylindricalPos'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CylindricalAcc)
    <class 'coordinax...CylindricalVel'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.SphericalVel)
    <class 'coordinax...SphericalPos'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.SphericalAcc)
    <class 'coordinax...SphericalVel'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.MathSphericalVel)
    <class 'coordinax...MathSphericalPos'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.MathSphericalAcc)
    <class 'coordinax...MathSphericalVel'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.LonLatSphericalVel)
    <class 'coordinax...LonLatSphericalPos'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.LonCosLatSphericalVel)
    <class 'coordinax...LonLatSphericalPos'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.LonLatSphericalAcc)
    <class 'coordinax...LonLatSphericalVel'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.ProlateSpheroidalVel)
    <class 'coordinax...ProlateSpheroidalPos'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.ProlateSpheroidalAcc)
    <class 'coordinax...ProlateSpheroidalVel'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CartesianVelND)
    <class 'coordinax...CartesianPosND'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CartesianAccND)
    <class 'coordinax...CartesianVelND'>

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def time_nth_derivative_vector_type(
    obj: Any, /, *, n: int
) -> "type[coordinax.vecs.AbstractVector]":
    """Return the corresponding time nth derivative vector type.

    Examples
    --------
    >>> import coordinax as cx

    >>> cx.vecs.RadialPos.time_nth_derivative_cls(n=0)
    <class 'coordinax...RadialPos'>

    >>> cx.vecs.RadialPos.time_nth_derivative_cls(n=1)
    <class 'coordinax...RadialVel'>

    >>> cx.vecs.RadialPos.time_nth_derivative_cls(n=2)
    <class 'coordinax...RadialAcc'>

    >>> cx.vecs.RadialVel.time_nth_derivative_cls(n=-1)
    <class 'coordinax...RadialPos'>

    >>> cx.vecs.RadialVel.time_nth_derivative_cls(n=0)
    <class 'coordinax...RadialVel'>

    >>> cx.vecs.RadialVel.time_nth_derivative_cls(n=1)
    <class 'coordinax...RadialAcc'>

    >>> cx.vecs.RadialAcc.time_nth_derivative_cls(n=-2)
    <class 'coordinax...RadialPos'>

    >>> cx.vecs.RadialAcc.time_nth_derivative_cls(n=-1)
    <class 'coordinax...RadialVel'>

    >>> cx.vecs.RadialAcc.time_nth_derivative_cls(n=0)
    <class 'coordinax...RadialAcc'>

    """
    raise NotImplementedError  # pragma: no cover
