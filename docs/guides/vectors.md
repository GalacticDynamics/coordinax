# Vectors

This guide covers the creation and use of vector objects in `coordinax`,
including positions, velocities, accelerations, arithmetic, dimensionality,
spaces, and vector functions.

## Creating Vector Objects

You can create vectors for positions, velocities, and accelerations in many
supported dimension:

```{code-block} python
>>> import coordinax.vecs as cxv
>>> q1 = cxv.CartesianPos1D.from_(1, "kpc")
>>> q2 = cxv.CartPos2D.from_([1, 2], "kpc")
>>> q3 = cxv.CartPos3D.from_([1, 2, 3], "kpc")
>>> v3 = cxv.CartVel3D.from_([4, 5, 6], "kpc/Myr")
>>> a3 = cxv.CartesianAcc3D.from_([0.1, 0.2, 0.3], "kpc/Myr^2")
```

You can also create N-D vectors:

```{code-block} python
>>> qn = cxv.CartPosND.from_([1, 2, 3, 4], "kpc")
```

All vector types support flexible input: scalars, lists, arrays, or
{class}`~unxt.quantity.Quantity`.

The component values can be multidimensional arrays, allowing for batch
operations:

```{code-block} python
>>> arr = cxv.CartPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")

```

## Arithmetic and Mathematical Operations

Vector objects support arithmetic and mathematical operations:

```{code-block} python
>>> q3 + q3
CartPos3D(x=Q(2, 'kpc'), y=Q(4, 'kpc'), z=Q(6, 'kpc'))

>>> 2 * q3
CartPos3D(x=Q(2, 'kpc'), y=Q(4, 'kpc'), z=Q(6, 'kpc'))

>>> v3 - v3
CartVel3D(x=Q(0, 'kpc / Myr'), y=Q(0, 'kpc / Myr'), z=Q(0, 'kpc / Myr'))
```

## Dimensionality: 1,N-D

`coordinax` provides vector classes for many dimensions:

- {class}`~coordinax.vecs.CartesianPos1D`, {class}`~coordinax.vecs.CartPos2D`,
  {class}`~coordinax.vecs.CartPos3D`, {class}`~coordinax.vecs.CartPosND`
- Similar classes for velocities (`CartVel*`), accelerations (`CartesianAcc*`),
  etc.
- Spacetime vectors {class}`~coordinax.vecs.FourVector`

## Conversion Between Representations

Vectors can be converted between coordinate systems:

```{code-block} python
>>> sph = q3.vconvert(cxv.SphericalPos)
>>> print(sph)
<SphericalPos: (r[kpc], theta[rad], phi[rad])
    [3.742 0.641 1.107]>
```

## Batch and Broadcast Operations

All vector types support batch operations and broadcasting:

```{code-block} python
>>> arr = cxv.CartPos3D.from_([[1,2,3],[4,5,6]], "kpc")
>>> arr * 2
CartPos3D(x=Q([2, 8], 'kpc'), y=Q([ 4, 10], 'kpc'),
               z=Q([ 6, 12], 'kpc'))
```

## Space Objects: Grouping Related Vectors

A {class}`~coordinax.vecs.KinematicSpace` object collects related vectors (e.g.,
position, velocity, acceleration):

```{code-block} python
>>> space = cxv.KinematicSpace(length=q3, speed=v3, acceleration=a3)
>>> print(space)
KinematicSpace({
    'length': <CartPos3D: (x, y, z) [kpc]
        [1 2 3]>,
    'speed': <CartVel3D: (x, y, z) [kpc / Myr]
        [4 5 6]>,
    'acceleration': <CartesianAcc3D: (x, y, z) [kpc / Myr2]
        [0.1 0.2 0.3]>
})
```

You can convert all vectors in a {class}`~coordinax.vecs.KinematicSpace` at
once:

```{code-block} python
>>> space_sph = space.vconvert(cxv.SphericalPos)
>>> print(space_sph.round(3))
KinematicSpace({
       'length':
       <SphericalPos: (r[kpc], theta[rad], phi[rad])
           [3.742 0.641 1.107]>,
       'speed':
       <SphericalVel: (r[kpc / Myr], theta[rad / Myr], phi[rad / Myr])
           [ 8.552  0.383 -0.6  ]>,
       'acceleration':
       <SphericalAcc: (r[kpc / Myr2], theta[rad / Myr2], phi[rad / Myr2])
           [ 0.374 -0.     0.   ]>
    })
```

---

:::{seealso}

[API Documentation for Vectors](../api/vecs.md)

:::
