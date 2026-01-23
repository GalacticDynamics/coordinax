# Coordinates and Frames

The {mod}`coordinax.frames` module provides a flexible and extensible framework
for working with reference frames and coordinate objects. This system allows you
to represent positions in different frames, transform between frames, and attach
frame metadata to your data.

## Built-in Frames

{mod}`coordinax.frames` includes several standard astronomical frames:

```{code-block} text
>>> import coordinax.frames as cxf
>>> icrs = cxf.ICRS()
>>> icrs
ICRS()

>>> gc = cxf.Galactocentric()
>>> gc
Galactocentric(
  galcen=LonLatSpherical3D(...),
  roll=Quantity(...),
  z_sun=Quantity(...),
  galcen_v_sun=Vector(...)
)
```

## Creating Coordinate Objects

Coordinate objects attach a vector (or {class}`~coordinax.PointedVector`) to a
frame:

```{code-block} text
>>> import coordinax as cx
>>> q = cx.Vector.from_([1, 2, 3], "kpc")
>>> coord = cxf.Coordinate(q, frame=icrs)
>>> print(coord)
Coordinate(
    {
        'base': <Cart3D: (x, y, z) [kpc]
            [1 2 3]>
    },
    frame=ICRS()
)
```

You can also create coordinates from an {class}`~coordinax.PointedVector` object
containing multiple vectors.

## Transforming Between Frames

You can transform coordinates or vectors between frames using transformation
operators:

```{code-block} text
>>> op = cxf.frame_transform_op(icrs, gc)
>>> q_gc = op(q)
>>> print(q_gc)
<Cart3D: (x, y, z) [kpc]
    [-11.375   1.845   0.133]>

>>> coord_gc = coord.to_frame(gc)
>>> print(coord_gc)
Coordinate(
    {
        'length': <Cart3D: (x, y, z) [kpc]
            [-11.375   1.845   0.133]>
    },
    frame=Galactocentric(...)
)
```

## Converting Representations Within a Frame

You can convert the internal representation of a coordinate (e.g., Cartesian to
Spherical) without changing its frame:

```{code-block} text
>>> coord_sph = coord.vconvert(cx.charts.sph3d)
>>> print(coord_sph)
Coordinate(
    {
        'length': <Spherical3D: (r[kpc], theta[rad], phi[rad])
            [3.742 0.641 1.107]>
    },
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
