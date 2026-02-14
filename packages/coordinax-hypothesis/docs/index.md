# coordinax-hypothesis

```{toctree}
:maxdepth: 1
:hidden:

api
testing-guide
```

Hypothesis strategies for property-based testing with
[coordinax](https://github.com/GalacticDynamics/coordinax).

This package provides [Hypothesis](https://hypothesis.readthedocs.io/)
strategies for generating random coordinax objects (angles, distances,
parallaxes, distance moduli) for property-based testing.

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install coordinax-hypothesis
```

:::

:::{tab-item} uv

```bash
uv add coordinax-hypothesis
```

:::

::::

## Quick Start

```
from hypothesis import given

import coordinax.angles as cxa
import coordinax.distances as cxd
import coordinax_hypothesis.core as cxst


@given(angle=cxst.angles())
def test_angle_property(angle):
    """Test that all angles are valid Angle objects."""
    assert isinstance(angle, cxa.Angle)


@given(dist=cxst.distances())
def test_distance_property(dist):
    """Test that all distances are valid Distance objects."""
    assert isinstance(dist, cxd.Distance)
    assert dist.value >= 0  # distances are non-negative by default
```

## Strategies

### `angles(wrap_to=None, **kwargs)`

Generate random {class}`unxt.Angle` objects for testing.

This strategy builds on {func}`unxt_hypothesis.quantities` to generate angles
with optional wrapping bounds. The strategy is useful for property-based testing
of angle-related computations.

**Parameters:**

- `wrap_to` (`SearchStrategy[tuple[Quantity, Quantity]] | None`): Optional
  hypothesis strategy that generates a tuple of (min_bound, max_bound) for angle
  wrapping. If None, generates angles without wrapping (default: None).
- `**kwargs`: Additional keyword arguments passed to
  {func}`~unxt_hypothesis.quantities`. Common options include:
  - `units` (str): Unit for the angle (e.g., "rad", "deg")
  - `min_value` (float): Minimum value for the angle
  - `max_value` (float): Maximum value for the angle
  - `shape` (int | tuple | SearchStrategy): Shape of the generated angle array

**Returns:** `SearchStrategy[unxt.Angle]`

**Examples:**

```
from hypothesis import given, strategies as st

import unxt as u
import coordinax_hypothesis.core as cxst


# Generate basic angles (no wrapping)
@given(angle=cxst.angles())
def test_basic_angle(angle):
    assert isinstance(angle, u.Angle)


# Generate angles in degrees
@given(angle=cxst.angles(units="deg"))
def test_angle_degrees(angle):
    assert angle.unit == "deg"


# Generate angles with wrapping bounds [0, 360) degrees
@given(angle=cxst.angles(wrap_to=st.just((u.Q(0, "deg"), u.Q(360, "deg")))))
def test_angle_with_wrapping(angle):
    assert angle.wrap_to is not None
    min_bound, max_bound = angle.wrap_to
    assert min_bound.to("deg").value == 0
    assert max_bound.to("deg").value == 360


# Generate angle arrays
@given(angle=cxst.angles(shape=10))
def test_angle_array(angle):
    assert angle.shape == (10,)


# Generate 2D angle arrays
@given(angle=cxst.angles(shape=(5, 3)))
def test_angle_2d(angle):
    assert angle.shape == (5, 3)
```

### `distances(check_negative=True, **kwargs)`

Generate random `coordinax.Distance` objects for testing.

This strategy builds on `unxt_hypothesis.quantities` to generate distances with
automatic handling of the non-negativity constraint. The strategy is useful for
property-based testing of distance-related computations.

**Parameters:**

- `check_negative` (bool | SearchStrategy[bool]): Whether to enforce
  non-negative distances. If `True` (default), generated distances will be >= 0.
  Can be a hypothesis strategy to vary this behavior across test examples.
- `**kwargs`: Additional keyword arguments passed to
  `unxt_hypothesis.quantities`. Common options include:
  - `units` (str): Unit for the distance (e.g., "kpc", "m", "AU")
  - `shape` (int | tuple | SearchStrategy): Shape of the generated distance
    array
  - `elements` (SearchStrategy): Strategy for generating array elements. When
    `check_negative=True`, the min_value will be automatically adjusted to 0 if
    needed.

**Returns:** `SearchStrategy[coordinax.Distance]`

**Examples:**

```
from hypothesis import given, strategies as st

import coordinax.distances as cxd
import coordinax_hypothesis.core as cxst


# Generate basic non-negative distances
@given(dist=cxst.distances())
def test_basic_distance(dist):
    assert isinstance(dist, cxd.Distance)
    assert dist.value >= 0


# Generate distances in specific units
@given(dist=cxst.distances(units="kpc"))
def test_distance_kpc(dist):
    assert dist.unit == "kpc"


# Allow negative distances
@given(dist=cxst.distances(check_negative=False))
def test_signed_distance(dist):
    assert isinstance(dist, cxd.Distance)
    # Can be positive or negative


# Generate distance arrays
@given(dist=cxst.distances(shape=10))
def test_distance_array(dist):
    assert dist.shape == (10,)
    assert all(dist.value >= 0)


# Control the value range
@given(dist=cxst.distances(elements=st.floats(min_value=10.0, max_value=100.0)))
def test_distance_range(dist):
    assert 10.0 <= dist.value <= 100.0
```

### `distance_moduli(**kwargs)`

Generate random `coordinax.DistanceModulus` objects for testing. distance and
angle strategiesategy builds on `unxt_hypothesis.quantities` to generate
distance moduli (apparent minus absolute magnitude). Distance moduli are always
in units of 'mag'. The strategy is useful for property-based testing of
magnitude-based distance computations.

**Parameters:**

- `**kwargs`: Additional keyword arguments passed to
  `unxt_hypothesis.quantities`. Common options include:
  - `shape` (int | tuple | SearchStrategy): Shape of the generated distance
    modulus array
  - `elements` (SearchStrategy): Strategy for generating array elements

**Returns:** `SearchStrategy[coordinax.DistanceModulus]`

**Examples:**

```
from hypothesis import given, strategies as st

import coordinax as cx
import coordinax_hypothesis.core as cxst


# Generate basic distance moduli
@given(dm=cxst.distance_moduli())
def test_basic_dm(dm):
    assert isinstance(dm, cxd.DistanceModulus)
    assert dm.unit == "mag"


# Generate distance modulus arrays
@given(dm=cxst.distance_moduli(shape=10))
def test_dm_array(dm):
    assert dm.shape == (10,)
    assert dm.unit == "mag"


# Control the value range (e.g., typical values 0-30 mag)
@given(dm=cxst.distance_moduli(elements=st.floats(min_value=0.0, max_value=30.0)))
def test_dm_range(dm):
    assert 0.0 <= dm.value <= 30.0
```

### `parallaxes(check_negative=True, **kwargs)`

Generate random `coordinax.Parallax` objects for testing.

This strategy builds on `unxt_hypothesis.quantities` to generate parallaxes with
automatic handling of the non-negativity constraint. While theoretically
parallax must be non-negative (tan(p) = 1 AU / d), noisy direct measurements can
yield negative values. The strategy is useful for property-based testing of
parallax-based distance computations.

**Parameters:**

- `check_negative` (bool | SearchStrategy[bool]): Whether to enforce
  non-negative parallaxes. If `True` (default), generated parallaxes will
  be >= 0. Can be a hypothesis strategy to vary this behavior across test
  examples. Set to `False` when testing handling of noisy measurements.
- `**kwargs`: Additional keyword arguments passed to
  {func}`unxt_hypothesis.quantities`. Common options include:
  - `units` (str): Unit for the parallax (e.g., "mas", "arcsec", "deg")
  - `shape` (int | tuple | SearchStrategy): Shape of the generated parallax
    array
  - `elements` (SearchStrategy): Strategy for generating array elements. When
    `check_negative=True`, the min_value will be automatically adjusted to 0 if
    needed.

**Returns:** `SearchStrategy[coordinax.Parallax]`

**Examples:**

```
from hypothesis import given, strategies as st

import coordinax as cx
import coordinax_hypothesis.core as cxst


# Generate basic non-negative parallaxes
@given(plx=cxst.parallaxes())
def test_basic_parallax(plx):
    assert isinstance(plx, cx.Parallax)
    assert plx.value >= 0


# Generate parallaxes in milliarcseconds
@given(plx=cxst.parallaxes(units="mas"))
def test_parallax_mas(plx):
    assert plx.unit == "mas"


# Allow negative parallaxes (for noisy measurements)
@given(plx=cxst.parallaxes(check_negative=False))
def test_noisy_parallax(plx):
    assert isinstance(plx, cx.Parallax)
    # Can be positive or negative


# Generate parallax arrays
@given(plx=cxst.parallaxes(shape=10))
def test_parallax_array(plx):
    assert plx.shape == (10,)
    assert all(plx.value >= 0)


# Control the value range (e.g., nearby stars with large parallax)
@given(
    plx=cxst.parallaxes(units="mas", elements=st.floats(min_value=1.0, max_value=100.0))
)
def test_nearby_parallax(plx):
    assert 1.0 <= plx.to("mas").value <= 100.0
```

### `charts_like(representation)`

Generate representations matching the flags of a template representation.

This strategy inspects a template representation to determine its type flags
(e.g., `Abstract1D`, `Abstract2D`, `Abstract3D`, `AbstractSpherical3D`, etc.)
and dimensionality, then generates new representations matching those same
criteria. This is particularly useful for generating varied test cases while
preserving key structural properties like dimensionality.

The strategy examines the template's method resolution order (MRO) to discover
all `AbstractDimensionalFlag` subclasses and uses them to filter the generated
representations.

**Parameters:**

- `representation` (AbstractChart | SearchStrategy): A representation instance
  to use as a template, or a strategy that generates one. The generated
  representations will match all the flags and dimensionality of the template.

**Returns:** `SearchStrategy[AbstractChart]`

**Examples:**

```

from hypothesis import given

import coordinax as cx
import coordinax_hypothesis.core as cxst


# Generate 3D representations like Cart3D
@given(chart=t=cxst.charts_like(cxc.cart3d))
def test_3d_chart(rep):
    assert isinstance(rep, cxc.Abstract3D)
    assert rep.ndim == 3
    # Could be Cart3D, Spherical3D, Cylindrical3D, etc.


# Generate 2D representations like Polar2D
@given(chart=t=cxst.charts_like(cxc.polar2d))
def test_2d_chart(rep):
    assert isinstance(rep, cxc.Abstract2D)
    assert rep.ndim == 2
    # Could be Cart2D, Polar2D, TwoSphere, etc.


# Generate 1D representations
@given(chart=t=cxst.charts_like(cxc.radial1d))
def test_1d_chart(rep):
    assert isinstance(rep, cxc.Abstract1D)
    assert rep.ndim == 1


# Use with a dynamic template
@given(chart=t=cxst.charts_like(cxst.charts(filter=cxc.Abstract3D)))
def test_charts_like(rep):
    # Will match the template's flags
    assert isinstance(rep, cxc.Abstract3D)
```

### `chart_time_chain(role, rep)`

Generate a chain of representations following the time antiderivative pattern.

Given a role flag (position, velocity, or acceleration) and a representation,
this strategy returns a tuple containing representations that match the flags of
each time antiderivative up to and including a position representation. Each
element in the chain is generated using `charts_like()` to match the flags of
the corresponding time antiderivative.

This is particularly useful for testing coordinate transformations across
different time derivatives (e.g., converting from acceleration to velocity to
position) while ensuring all representations in the chain share compatible
flags.

**Parameters:**

- `role` (AbstractRole): The starting role (`cxr.PhysDispsDisp`, `cxr.PhysVel`,
  `cxr.PhysAcc`).
- `rep` (AbstractChart | SearchStrategy): The starting representation or a
  strategy that generates one.

**Returns:** `SearchStrategy[tuple[AbstractChart, ...]]`

The returned tuple follows this pattern:

- If input is position: `(Point,)`
- If input is velocity: `(PhysVel, Point)`
- If input is acceleration: `(PhysAcc, PhysVel, Point)`

**Examples:**

```
from hypothesis import given

import coordinax as cx
import coordinax_hypothesis.core as cxst


# Generate a chain from acceleration
@given(chain=cxst.chart_time_chain(cxr.PhysAcc, cxc.cart3d))
def test_acc_chain(chain):
    acc_chart, vel_chart, point_chart = chain
    # All are 3D Cartesian-like representations
    assert isinstance(acc_chart, cxc.Abstract3D)
    assert isinstance(vel_chart, cxc.Abstract3D)
    assert isinstance(point_chart, cxc.Abstract3D)


# Generate a chain from velocity
@given(chain=cxst.chart_time_chain(cxr.PhysVel, cxc.polar2d))
def test_vel_chain(chain):
    vel_chart, point_chart = chain
    # All are 2D representations
    assert isinstance(vel_chart, cxc.Abstract2D)
    assert isinstance(point_chart, cxc.Abstract2D)


# Position just returns itself
@given(chain=cxst.chart_time_chain(cxr.PhysDisp, cxc.sph3d))
def test_disp_chain(chain):
    (point_chart,) = chain
    assert isinstance(point_chart, cxc.Abstract3D)


# Use with dynamic representation type
@given(chain=cxst.chart_time_chain(cxr.PhysVel, cxst.charts()))
def test_dynamic_vel_chain(chain):
    assert len(chain) == 2  # (vel, point)
    assert isinstance(chain[0], cxc.AbstractChart)
    assert isinstance(chain[1], cxc.AbstractChart)
```

### `vectors_with_target_chart(chart=t=charts(), role=cxr.PhysDisp, dtype=jnp.float32, shape=(), elements=None)`

Generate a vector and a time-derivative chain with matching flags.

This strategy is useful for testing conversion operations where you need a
source vector and a full set of target representations (following the time
antiderivative chain) that it can be converted to. The source vector and all
target representations will have compatible dimensionalities and flags. The
target chain automatically matches the flags of the source vector.

**Parameters:**

- `rep` (AbstractChart | SearchStrategy): A representation instance or strategy
  for the source vector (default: uses `charts()` strategy).
- `role` (AbstractRole): The role flag for the source vector (`cxr.PhysDisp`,
  `cxr.PhysVel`, `cxr.PhysAcc`).
- `dtype` (dtype | SearchStrategy): The data type for array components (default:
  `jnp.float32`). Can be a dtype or a strategy.
- `shape` (int | tuple | SearchStrategy): The shape for the vector components
  (default: scalar shape `()`).
- `elements` (SearchStrategy, optional): Strategy for generating element values.
  If None, uses finite floats.

**Returns:** `SearchStrategy[tuple[Vector, tuple[AbstractChart, ...]]]`

A tuple of `(vector, target_chain)` where `target_chain` is a tuple of
representations following the time antiderivative pattern, all matching the
flags of the source vector's representation.

**Examples:**

```
from hypothesis import given

import coordinax as cx
import coordinax_hypothesis.core as cxst


# Test vector conversions to a full chain of targets
@given(vec_and_chain=cxst.vectors_with_target_chart(chart=t=cxc.cart3d, role=cxr.PhysDisp))
def test_position_conversion(vec_and_chain):
    vec, target_chain = vec_and_chain
    # target_chain is just (pos_chart,) for position sources
    (target_chart,) = target_chain
    converted = vec.vconvert(target_chart)
    assert converted.chart == target_chart


# Test velocity vector with full chain (requires a position vector)
@given(
    vec_and_chain=cxst.vectors_with_target_chart(chart=t=cxc.cart3d, role=cxr.PhysVel),
    pos_vec=cxst.vectors(chart=t=cxc.cart3d, role=cxr.PhysDisp),
)
def test_velocity_conversion_chain(vec_and_chain, pos_vec):
    vec, target_chain = vec_and_chain
    # target_chain is (vel_chart, pos_chart)
    for target_chart in target_chain:
        converted = vec.vconvert(target_chart, pos_vec)
        assert converted.chart == target_chart


# Test with specific dimensionality
@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=t=cxst.charts(filter=cxc.Abstract3D),
        role=cxr.PhysDisp,
    )
)
def test_3d_position_conversions(vec_and_chain):
    vec, (target_chart,) = vec_and_chain
    assert isinstance(vec.chart, cxc.Abstract3D)
    assert isinstance(target_chart, cxc.Abstract3D)
    converted = vec.vconvert(target_chart)
    assert converted.chart == target_chart
```

## Integration with `unxt-hypothesis`

The {mod}`coordinax-hypothesis` package builds on top of {mod}`unxt-hypothesis`
strategies.

For more advanced usage patterns, see the [Testing Guide](testing-guide.md).

## Contributing

Contributions are welcome! Please see the
[coordinax contributing guide](https://github.com/GalacticDynamics/coordinax/blob/main/CONTRIBUTING.md)
for details.

## License

`coordinax-hypothesis` is licensed under the MIT License. See the LICENSE file
for details.
