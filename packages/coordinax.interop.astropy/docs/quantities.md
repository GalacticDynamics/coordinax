# Quantities

<!-- invisible-code-block: python
import importlib.util
-->
<!-- skip: start if(importlib.util.find_spec('coordinax.interop.astropy') is None, reason="coordinax.interop.astropy not installed") -->

This guide demonstrates how to convert quantity-like objects between {mod}`coordinax` and {mod}`astropy` using {mod}`plum` dispatch and `from_` constructors.

```{code-block} python
>>> import jax.numpy as jnp
>>> import plum

>>> import coordinax.main as cx

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

Convert {mod}`coordinax` to {mod}`astropy`:

```{code-block} python
>>> plum.convert(angle, apyu.Quantity)
<Quantity [1., 2., 3.] rad>

>>> angle_apy = plum.convert(angle, apyc.Angle)
>>> angle_apy
<Angle [1., 2., 3.] rad>
```

Convert {mod}`astropy` back to {mod}`coordinax`:

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

Convert {mod}`coordinax` to {mod}`astropy`:

```{code-block} python
>>> plum.convert(distance, apyu.Quantity)
<Quantity [1., 2., 3.] km>

>>> distance_apy = plum.convert(distance, apyc.Distance)
>>> distance_apy
<Distance [1., 2., 3.] km>
```

Convert {mod}`astropy` back to {mod}`coordinax`:

```{code-block} python
>>> plum.convert(distance_apy, cx.Distance)
Distance([1., 2., 3.], 'km')

>>> cx.Distance.from_(distance_apy)
Distance([1., 2., 3.], 'km')
```

## Distance modulus conversions

Create a {class}`coordinax.distances.DistanceModulus`:

```{code-block} python
>>> distmod = cx.DistanceModulus(jnp.array([1, 2, 3]), "mag")
>>> distmod
DistanceModulus([1, 2, 3], 'mag')
```

Convert {mod}`coordinax` to {mod}`astropy`:

```{code-block} python
>>> distmod_apy = plum.convert(distmod, apyu.Quantity)
>>> distmod_apy
<Quantity [1., 2., 3.] mag>
```

Convert {mod}`astropy` back to {mod}`coordinax`:

```{code-block} python
>>> plum.convert(distmod_apy, cx.DistanceModulus)
DistanceModulus([1., 2., 3.], 'mag')

>>> cx.DistanceModulus.from_(distmod_apy)
DistanceModulus([1., 2., 3.], 'mag')
```

## Parallax conversions

Create a {class}`coordinax.distances.Parallax`:

```{code-block} python
>>> parallax = cx.Parallax(jnp.array([1, 2, 3]), "rad")
>>> parallax
Parallax([1, 2, 3], 'rad')
```

Convert {mod}`coordinax` to {mod}`astropy`:

```{code-block} python
>>> parallax_apy = plum.convert(parallax, apyu.Quantity)
>>> parallax_apy
<Quantity [1., 2., 3.] rad>
```

Convert {mod}`astropy` back to {mod}`coordinax`:

```{code-block} python
>>> plum.convert(parallax_apy, cx.Parallax)
Parallax([1., 2., 3.], 'rad')

>>> cx.Parallax.from_(parallax_apy)
Parallax([1., 2., 3.], 'rad')
```
