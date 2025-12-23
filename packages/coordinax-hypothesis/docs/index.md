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

```python
from hypothesis import given

import coordinax as cx
import coordinax_hypothesis as cxst


@given(angle=cxst.angles())
def test_angle_property(angle):
    """Test that all angles are valid Angle objects."""
    assert isinstance(angle, cx.Angle)


@given(dist=cxst.distances())
def test_distance_property(dist):
    """Test that all distances are valid Distance objects."""
    assert isinstance(dist, cx.Distance)
    assert dist.value >= 0  # distances are non-negative by default
```

## Strategies

### `angles(wrap_to=None, **kwargs)`

Generate random `unxt.Angle` objects for testing.

This strategy builds on `unxt_hypothesis.quantities` to generate angles with
optional wrapping bounds. The strategy is useful for property-based testing of
angle-related computations.

**Parameters:**

- `wrap_to` (SearchStrategy[tuple[Quantity, Quantity]] | None): Optional
  hypothesis strategy that generates a tuple of (min_bound, max_bound) for angle
  wrapping. If None, generates angles without wrapping (default: None).
- `**kwargs`: Additional keyword arguments passed to
  `unxt_hypothesis.quantities`. Common options include:
  - `units` (str): Unit for the angle (e.g., "rad", "deg")
  - `min_value` (float): Minimum value for the angle
  - `max_value` (float): Maximum value for the angle
  - `shape` (int | tuple | SearchStrategy): Shape of the generated angle array

**Returns:** `SearchStrategy[unxt.Angle]`

**Examples:**

```python
from hypothesis import given, strategies as st

import unxt as u
import coordinax_hypothesis as cxst


# Generate basic angles (no wrapping)
@given(angle=cxst.angles())
def test_basic_angle(angle):
    assert isinstance(angle, u.Angle)


# Generate angles in degrees
@given(angle=cxst.angles(units="deg"))
def test_angle_degrees(angle):
    assert angle.unit == "deg"


# Generate angles with wrapping bounds [0, 360) degrees
@given(
    angle=cxst.angles(wrap_to=st.just((u.Quantity(0, "deg"), u.Quantity(360, "deg"))))
)
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

```python
from hypothesis import given, strategies as st

import coordinax as cx
import coordinax_hypothesis as cxst


# Generate basic non-negative distances
@given(dist=cxst.distances())
def test_basic_distance(dist):
    assert isinstance(dist, cx.Distance)
    assert dist.value >= 0


# Generate distances in specific units
@given(dist=cxst.distances(units="kpc"))
def test_distance_kpc(dist):
    assert dist.unit == "kpc"


# Allow negative distances
@given(dist=cxst.distances(check_negative=False))
def test_signed_distance(dist):
    assert isinstance(dist, cx.Distance)
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

```python
from hypothesis import given, strategies as st

import coordinax as cx
import coordinax_hypothesis as cxst


# Generate basic distance moduli
@given(dm=cxst.distance_moduli())
def test_basic_dm(dm):
    assert isinstance(dm, cx.DistanceModulus)
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
  `unxt_hypothesis.quantities`. Common options include:
  - `units` (str): Unit for the parallax (e.g., "mas", "arcsec", "deg")
  - `shape` (int | tuple | SearchStrategy): Shape of the generated parallax
    array
  - `elements` (SearchStrategy): Strategy for generating array elements. When
    `check_negative=True`, the min_value will be automatically adjusted to 0 if
    needed.

**Returns:** `SearchStrategy[coordinax.Parallax]`

**Examples:**

```python
from hypothesis import given, strategies as st

import coordinax as cx
import coordinax_hypothesis as cxst


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

## Integration with unxt-hypothesis

The `coordinax-hypothesis` package builds on top of
[unxt-hypothesis](https://github.com/GalacticDynamics/unxt) strategies. All
`unxt_hypothesis.quantities` parameters can be passed through the `angles`
strategy.

For more advanced usage patterns, see the [Testing Guide](testing-guide.md).

## Contributing

Contributions are welcome! Please see the
[coordinax contributing guide](https://github.com/GalacticDynamics/coordinax/blob/main/CONTRIBUTING.md)
for details.

## License

`coordinax-hypothesis` is licensed under the MIT License. See the LICENSE file
for details.
