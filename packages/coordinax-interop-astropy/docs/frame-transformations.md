# Frame Transformations

<!-- invisible-code-block: python
import importlib.util
-->

This guide demonstrates how to use {mod}`coordinax-interop-astropy` to work with
astronomical reference frames and perform transformations between them.

## ICRS to Galactocentric Transformation

This example shows how to transform coordinates from the
{class}`~coordinax_astro.ICRS` (International Celestial Reference System) frame
to a {class}`~coordinax_astro.Galactocentric` frame. We'll use Vega as our test
star and compare against Astropy's implementation to verify accuracy.

### Setting up the example

<!-- skip: next if(importlib.util.find_spec('coordinax_interop_astropy') is None, reason="coordinax-interop-astropy not installed") -->

First, let's define Vega's position and velocity in ICRS coordinates using
{mod}`astropy`:

```python
import unxt as u
import coordinax as cx
import coordinax.vecs as cxv
import astropy.coordinates as apyc

# The location of Vega in ICRS coordinates
vega = apyc.SkyCoord(
    ra=279.23473479 * u.unit("deg"),
    dec=38.78368896 * u.unit("deg"),
    distance=25 * u.unit("pc"),
    pm_ra_cosdec=200 * u.unit("mas / yr"),
    pm_dec=-286 * u.unit("mas / yr"),
    radial_velocity=-13.9 * u.unit("km / s"),
)
print(vega)
# <SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, pc)
#     (279.23473479, 38.78368896, 25.)
#  (pm_ra_cosdec, pm_dec, radial_velocity) in (mas / yr, mas / yr, km / s)
#     (200., -286., -13.9)>
```

### Astropy transformation for comparison

Let's first see what {mod}`astropy` gives us:

```python
# Transforming to a Galactocentric frame
apy_gcf = apyc.Galactocentric()
apy_gcf
# <Galactocentric Frame (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
#     (266.4051, -28.936175)>, galcen_distance=8.122 kpc, galcen_v_sun=(12.9, 245.6, 7.78) km / s, z_sun=20.8 pc, roll=0.0 deg)>

vega.transform_to(apy_gcf)
# <SkyCoord (Galactocentric: ...): (x, y, z) in pc
#     (-8112.89970167, 21.79911216, 29.01384942)
#  (v_x, v_y, v_z) in km / s
#     (34.06711868, 234.61647066, -28.75976702)>
```

### Converting to coordinax

Now we can use `coordinax` to perform the same transformation! First, convert
the {mod}`astropy` objects to {mod}`coordinax`:

```python
vega_q = cxv.LonLatSphericalPos.from_(vega.icrs.data)
vega_p = cxv.LonCosLatSphericalVel.from_(vega.icrs.data.differentials["s"])

icrs_frame = cx.frames.ICRS()
gcf_frame = cx.frames.Galactocentric.from_(apy_gcf)
```

### Defining and applying the transformation

Define the transformation operator:

```python
frame_op = cx.frames.frame_transform_op(icrs_frame, gcf_frame)
frame_op
# Pipe((
#     GalileanRotation(rotation=f32[3,3]),
#     GalileanSpatialTranslation(CartesianPos3D( ... )),
#     GalileanRotation(rotation=f32[3,3]),
#     VelocityBoost(CartesianVel3D( ... ))
# ))
```

Apply the transformation:

```python
vega_gcf_q, vega_gcf_p = frame_op(vega_q, vega_p)
vega_gcf_q = vega_gcf_q.vconvert(cxv.CartesianPos3D)
vega_gcf_p = vega_gcf_p.vconvert(cxv.CartesianVel3D, vega_gcf_q)
print(vega_gcf_q)
# <CartesianPos3D: (x, y, z) [pc]
#     [-8112.898    21.799    29.01...]>
print(vega_gcf_p.uconvert({u.dimension("speed"): "km/s"}))
# <CartesianVel3D: (x, y, z) [km / s]
#     [ 34.067 234.616 -28.76 ]>
```

It matches!

### Working with different input types

The transformation operators are flexible and work with various input types:

#### With unxt Quantities

```python
q = u.Q([0, 0, 0], "pc")
frame_op(q)
# Quantity(Array([-8121.972, 0. , 20.8 ], dtype=float32), unit='pc')

p = u.Q([0.0, 0, 0], "km/s")

newq, newp = frame_op(q, p)
newq, newp
# (Quantity(Array([-8121.972,     0.   ,    20.8  ], dtype=float32), unit='pc'),
#  Quantity(Array([ 12.9 , 245.6 ,   7.78], dtype=float32), unit='km / s'))
```

#### With coordinax vectors

```python
q = cx.CartesianPos3D.from_([0, 0, 0], "pc")
p = cx.CartesianVel3D.from_([0, 0, 0], "km/s")

newq, newp = frame_op(q, p)
print(newq, newp, sep="\n")
# <CartesianPos3D: (x, y, z) [pc]
#     [-8121.972     0.       20.8  ]>
# <CartesianVel3D: (x, y, z) [km / s]
#     [ 12.9  245.6    7.78]>
```

## Galactocentric to ICRS Transformation

The reverse transformation from {mod}`~coordinax_astro.Galactocentric` to
{mod}`~coordinax_astro.ICRS` is just as easy:

```python
# Define transformation operator in reverse direction
frame_op_reverse = cx.frames.frame_transform_op(gcf_frame, icrs_frame)

# Starting from Galactocentric coordinates
vega_gcf = apyc.SkyCoord(
    x=-8112.89970167 * u.unit("pc"),
    y=21.79911216 * u.unit("pc"),
    z=29.01384942 * u.unit("pc"),
    v_x=34.06711868 * u.unit("km/s"),
    v_y=234.61647066 * u.unit("km/s"),
    v_z=-28.75976702 * u.unit("km/s"),
    frame=apy_gcf,
)

# Convert to coordinax
vega_gcf_q = cx.CartesianPos3D.from_(vega_gcf.galactocentric.data)
vega_gcf_p = cx.CartesianVel3D.from_(vega_gcf.galactocentric.data.differentials["s"])

# Apply transformation
vega_icrs_q, vega_icrs_p = frame_op_reverse(vega_gcf_q, vega_gcf_p)

# Convert to spherical coordinates for comparison
vega_icrs_q = vega_icrs_q.vconvert(cx.vecs.LonLatSphericalPos)
vega_icrs_p = vega_icrs_p.vconvert(cx.vecs.LonCosLatSphericalVel, vega_icrs_q)
print(vega_icrs_q.uconvert({u.dimension("angle"): "deg", u.dimension("length"): "pc"}))
# <LonLatSphericalPos: (lon[deg], lat[deg], distance[pc])
#     [279.235  38.784  25.   ]>
print(
    vega_icrs_p.uconvert(
        {u.dimension("angular speed"): "mas / yr", u.dimension("speed"): "km/s"}
    )
)
# <LonCosLatSphericalVel: (lon_coslat[mas / yr], lat[mas / yr], distance[km / s])
#     [ 200.001 -286.  -13.9  ]>
```

Perfect agreement with the original coordinates!

## Custom Frame Parameters

You can create {class}`~coordinax_astro.Galactocentric` frames with custom
parameters:

```python
import coordinax_astro as cxa

# Custom galactocentric frame
custom_gcf = cxa.Galactocentric(
    galcen=cxv.LonLatSphericalPos(
        lon=u.Q(266.4, "deg"), lat=u.Q(-28.9, "deg"), distance=u.Q(8.5, "kpc")
    ),
    z_sun=u.Q(25, "pc"),
    roll=u.Q(5, "deg"),
    galcen_v_sun=cxv.CartesianVel3D(
        x=u.Q(12, "km/s"), y=u.Q(250, "km/s"), z=u.Q(8, "km/s")
    ),
)

# Create transformation with custom frame
custom_op = cx.frames.frame_transform_op(icrs_frame, custom_gcf)
```

## See Also

- [API Reference](api.md) for detailed function signatures
- [coordinax-astro documentation](https://coordinax.readthedocs.io/en/latest/packages/coordinax-astro/)
  for frame definitions
- [Astropy coordinates documentation](https://docs.astropy.org/en/stable/coordinates/)
  for background on astronomical coordinate systems
