# Quantities

<!-- invisible-code-block: python
import importlib.util
-->
<!-- skip: start if(importlib.util.find_spec('coordinax.interop.astropy') is None, reason="coordinax.interop.astropy not installed") -->

This guide demonstrates how to convert quantity-like objects between `coordinax` and `astropy` using `plum` dispatch and `from_` constructors.

```{code-block} python
>>> import jax.numpy as jnp
>>> import plum

>>> import coordinax.main as cx
>>> import coordinax.astro as cxastro

>>> import astropy.coordinates as apyc
>>> import astropy.units as apyu
```

## Angle conversions

Create a {class}`coordinax.angles.Angle`:

```{code-block} python
>>> angle = cx.Angle(jnp.array([1, 2, 3]), "rad")
>>> angle
Angle([1, 2, 3], 'rad')
```

Convert `coordinax` to `astropy`:

```{code-block} python
>>> plum.convert(angle, apyu.Quantity)
<Quantity [1., 2., 3.] rad>

>>> angle_apy = plum.convert(angle, apyc.Angle)
>>> angle_apy
<Angle [1., 2., 3.] rad>
```

Convert `astropy` back to `coordinax`:

```{code-block} python
>>> plum.convert(angle_apy, cx.Angle)
Angle([1., 2., 3.], 'rad')

>>> cx.Angle.from_(angle_apy)
Angle([1., 2., 3.], 'rad')
```

## Distance conversions

Create a {class}`coordinax.distances.Distance`:

```{code-block} python
>>> distance = cx.Distance(jnp.array([1, 2, 3]), "km")
>>> distance
Distance([1, 2, 3], 'km')
```

Convert `coordinax` to `astropy`:

```{code-block} python
>>> plum.convert(distance, apyu.Quantity)
<Quantity [1., 2., 3.] km>

>>> distance_apy = plum.convert(distance, apyc.Distance)
>>> distance_apy
<Distance [1., 2., 3.] km>
```

Convert `astropy` back to `coordinax`:

```{code-block} python
>>> plum.convert(distance_apy, cx.Distance)
Distance([1., 2., 3.], 'km')

>>> cx.Distance.from_(distance_apy)
Distance([1., 2., 3.], 'km')
```

## Distance modulus conversions

Create a {class}`coordinax.astro.DistanceModulus`:

```{code-block} python
>>> distmod = cxastro.DistanceModulus(jnp.array([1, 2, 3]), "mag")
>>> distmod
DistanceModulus([1, 2, 3], 'mag')
```

Convert `coordinax` to `astropy`:

```{code-block} python
>>> distmod_apy = plum.convert(distmod, apyu.Quantity)
>>> distmod_apy
<Quantity [1., 2., 3.] mag>
```

Convert `astropy` back to `coordinax`:

```{code-block} python
>>> plum.convert(distmod_apy, cxastro.DistanceModulus)
DistanceModulus([1., 2., 3.], 'mag')

>>> cxastro.DistanceModulus.from_(distmod_apy)
DistanceModulus([1., 2., 3.], 'mag')
```

## Parallax conversions

Create a {class}`coordinax.astro.Parallax`:

```{code-block} python
>>> parallax = cxastro.Parallax(jnp.array([1, 2, 3]), "rad")
>>> parallax
Parallax([1, 2, 3], 'rad')
```

Convert `coordinax` to `astropy`:

```{code-block} python
>>> parallax_apy = plum.convert(parallax, apyu.Quantity)
>>> parallax_apy
<Quantity [1., 2., 3.] rad>
```

Convert `astropy` back to `coordinax`:

```{code-block} python
>>> plum.convert(parallax_apy, cxastro.Parallax)
Parallax([1., 2., 3.], 'rad')

>>> cxastro.Parallax.from_(parallax_apy)
Parallax([1., 2., 3.], 'rad')
```
