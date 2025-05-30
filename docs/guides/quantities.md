# Specialized Quantity Objects

## Working with `Angle` Objects

The {class}`~coordinax.angle.Angle` class in {mod}`coordinax.angle` is a
specialized quantity for representing angular measurements, similar to
{class}`~unxt.quantity.Quantity` but with additional features and constraints
tailored for angles.

<!-- invisible-code-block: python

import unxt as u
import jax.numpy as jnp

-->

### Creating Angles

You can create an {class}`~coordinax.angle.Angle` just like a
{class}`~unxt.quantity.Quantity`, by specifying a value and a unit with angular
dimensions:

```{code-block} python
>>> import coordinax.angle as cxa
>>> a = cxa.Angle(45, "deg")
>>> a
Angle(Array(45, dtype=int32, weak_type=True), unit='deg')
```

Just like {class}`~unxt.quantity.Quantity`, you can flexibly create
{class}`~coordinax.angle.Angle` objects using the
{meth}`~coordinax.angle.Angle.from_` constructor:

```{code-block} python
>>> cxa.Angle.from_(45, "deg")
Angle(Array(45, dtype=int32, weak_type=True), unit='deg')

>>> cxa.Angle.from_([45, 90], "deg")
Angle(Array([45, 90], dtype=int32), unit='deg')

>>> cxa.Angle.from_(jnp.array([10, 15, 20]), "deg")
Angle(Array([10, 15, 20], dtype=int32), unit='deg')

```

### Mathematical Operations

{class}`~coordinax.angle.Angle` objects support arithmetic operations,
broadcasting, and most mathematical functions, just like
{class}`~unxt.quantity.Quantity`:

```{code-block} python
>>> b = cxa.Angle(30, "deg")
>>> a + b
Angle(Array(75, dtype=int32, weak_type=True), unit='deg')
>>> 2 * a
Angle(Array(90, dtype=int32, weak_type=True), unit='deg')
>>> a.to("rad")
Angle(Array(0.7853982, dtype=float32, weak_type=True), unit='rad')
```

For more information on mathematical operations, see the unxt documentation.

### Enforced Dimensionality

Unlike a generic {class}`~unxt.quantity.Quantity`, the
{class}`~coordinax.angle.Angle` class enforces that the unit must be angular
(e.g., degrees, radians). Attempting to use a non-angular unit will raise an
error:

```{code-block} python
>>> try: cxa.Angle(1, "m")
... except ValueError as e: print(e)
Angle must have units with angular dimensions.
```

### Wrapping Angles

A key feature of {class}`~coordinax.angle.Angle` is the ability to wrap values
to a specified range, which is useful for keeping angles within a branch cut:

```{code-block} python
>>> from unxt import Quantity
>>> a = cxa.Angle(370, "deg")
>>> a.wrap_to(Quantity(0, "deg"), Quantity(360, "deg"))
Angle(Array(10, dtype=int32, weak_type=True), unit='deg')
```

The {meth}`~coordinax.angle.Angle.wrap_to` method has a function counterpart

```{code-block} python
>>> cxa.wrap_to(a, u.Quantity(0, "deg"), u.Quantity(360, "deg"))
Angle(Array(10, dtype=int32, weak_type=True), unit='deg')
```

## Working with `Distance` Objects

The {class}`~coordinax.distance.Distance` class in {mod}`coordinax.distance` is
a specialized quantity for representing physical distances, with enforced
dimensionality and convenient conversions to and from other distance-like
representations. Related classes, {class}`~coordinax.distance.Parallax` and
{class}`~coordinax.distance.DistanceModulus`, are also provided for common
astronomical use cases.

### Creating Distance Objects

You can create a {class}`~coordinax.distance.Distance` just like a
{class}`~unxt.quantity.Quantity`, by specifying a value and a unit with length
dimensions:

```{code-block} python
>>> import coordinax.distance as cxd
>>> d = cxd.Distance(10, "kpc")
>>> d
Distance(Array(10, dtype=int32, weak_type=True), unit='kpc')
```

### Creating Parallax and DistanceModulus Objects

{class}`~coordinax.distance.Parallax` and
{class}`~coordinax.distance.DistanceModulus` are alternative representations of
distance:

```{code-block} python
>>> p = cxd.Parallax(0.1, "mas")
>>> p
Parallax(Array(0.1, dtype=float32, weak_type=True), unit='mas')

>>> dm = cxd.DistanceModulus(15, "mag")
>>> dm
DistanceModulus(Array(15, dtype=int32, weak_type=True), unit='mag')
```

### Properties and Conversions

Each of these classes provides properties to convert between representations:

```{code-block} python
>>> d.parallax
Parallax(Array(4.848137e-10, dtype=float32, weak_type=True), unit='rad')
>>> d.distance_modulus
DistanceModulus(Array(15., dtype=float32), unit='mag')

>>> p.distance.uconvert("kpc")
Distance(Array(10., dtype=float32, weak_type=True), unit='kpc')
>>> p.distance_modulus
DistanceModulus(Array(15., dtype=float32), unit='mag')

>>> dm.distance.uconvert("kpc")
Distance(Array(10., dtype=float32, weak_type=True), unit='kpc')
>>> dm.parallax
Parallax(Array(4.848137e-10, dtype=float32, weak_type=True), unit='rad')
```

All three classes enforce that their units are appropriate for their physical
meaning (e.g., {class}`~coordinax.distance.Distance` must have length units,
{class}`~coordinax.distance.Parallax` must have angular units, and
{class}`~coordinax.distance.DistanceModulus` must have magnitude units).

---

:::{seealso}

[API Documentation for Angles](../api/angle.md)

[API Documentation for Distances](../api/distance.md)

:::
