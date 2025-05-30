# Coordinates and Frames

The {mod}`coordinax.frames` module provides a flexible and extensible framework
for working with reference frames and coordinate objects. This system allows you
to represent positions in different frames, transform between frames, and attach
frame metadata to your data.

## Built-in Frames

{mod}`coordinax.frames` includes several standard astronomical frames:

```{code-block} python
>>> import coordinax.frames as cxf
>>> icrs = cxf.ICRS()
>>> icrs
ICRS()

>>> gc = cxf.Galactocentric()
>>> gc
Galactocentric(
  galcen=LonLatSphericalPos(...),
  roll=Quantity(...),
  z_sun=Quantity(...),
  galcen_v_sun=CartesianVel3D(...)
)
```

## Creating Coordinate Objects

Coordinate objects attach a vector (or {class}`~coordinax.vecs.Space`) to a
frame:

```{code-block} python
>>> import coordinax.vecs as cxv
>>> q = cxv.CartesianPos3D.from_([1, 2, 3], "kpc")
>>> coord = cxf.Coordinate(q, frame=icrs)
>>> print(coord)
Coordinate(
    data=Space({
        'length': <CartesianPos3D: (x, y, z) [kpc]
            [1 2 3]>
    }),
    frame=ICRS()
)
```

You can also create coordinates from a {class}`~coordinax.vecs.Space` object
containing multiple vectors.

## Transforming Between Frames

You can transform coordinates or vectors between frames using transformation
operators:

```{code-block} python
>>> op = cxf.frame_transform_op(icrs, gc)
>>> q_gc = op(q)
>>> print(q_gc)
<CartesianPos3D: (x, y, z) [kpc]
    [-11.375   1.845   0.133]>

>>> coord_gc = coord.to_frame(gc)
>>> print(coord_gc)
Coordinate(
    data=Space({
        'length': <CartesianPos3D: (x, y, z) [kpc]
                                            [-11.375   1.845   0.133]>
    }),
    frame=Galactocentric(...)
)
```

## Converting Representations Within a Frame

You can convert the internal representation of a coordinate (e.g., Cartesian to
Spherical) without changing its frame:

```{code-block} python
>>> coord_sph = coord.vconvert(cxv.SphericalPos)
>>> print(coord_sph)
Coordinate(
    data=Space({
        'length': <SphericalPos: (r[kpc], theta[rad], phi[rad])
            [3.742 0.641 1.107]>
    }),
    frame=ICRS()
)
```

## Custom Frames and Advanced Usage

You can define your own frames by subclassing the frame base classes and
specifying transformation logic. All features above apply to custom frames as
well.

---

:::{seealso}

[API Documentation for Coordinates and Frames](../api/frames.md)

:::
