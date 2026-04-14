# Testing with `coordinax.hypothesis`

This guide shows how to use the `coordinax.hypothesis` package for property-based testing of code that uses coordinax angles.

## What is Property-Based Testing?

Property-based testing is a testing methodology where you specify properties that should hold true for all inputs, and the testing framework (Hypothesis) generates random test cases to verify those properties.

Instead of writing:

```python
import unxt as u


def test_angle_normalization():
    angle = u.Angle(370, "deg")
    assert angle.to("deg").value == 370
```

You write:

```python
from hypothesis import given
import coordinax.hypothesis.main as cxst


@given(angle=cxst.angles(units="deg"))
def test_angle_properties(angle):
    """All angles have valid values and units."""
    assert isinstance(angle, u.Angle)
    assert angle.value is not None
    assert angle.unit is not None
```

Hypothesis will generate random test cases with different values, uncovering edge cases you might not have thought of.

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install coordinax.hypothesis
```

:::

:::{tab-item} uv

```bash
uv add coordinax.hypothesis
```

:::

::::

## Basic Examples

```python
from hypothesis import given, assume, strategies as st
import coordinax.hypothesis.main as cxst
import unxt as u
import jax.numpy as jnp
```

### Testing Angle Properties

```python
@given(angle=cxst.angles())  # any Angle
def test_angle_has_value_and_unit(angle):
    """Every Angle has a value and a unit."""
    assert angle.value is not None
    assert angle.unit is not None


@given(angle=cxst.angles(units="deg"))
def test_angle_in_degrees(angle):
    """Angles can be specified in degrees."""
    assert angle.unit == u.unit("deg")


@given(angle=cxst.angles(shape=(3,)))
def test_vector_angle_has_correct_shape(angle):
    """Vector angles have the expected shape."""
    assert angle.shape == (3,)
```

### Testing Angle Wrapping

```python
@given(angle=cxst.angles(wrap_to=st.just((u.Q(0, "deg"), u.Q(360, "deg")))))
def test_angle_with_wrapping_bounds(angle):
    """Angles with wrapping have valid wrap_to bounds."""
    assert angle.wrap_to is not None
    min_bound, max_bound = angle.wrap_to
    assert min_bound < max_bound


@given(angle=cxst.angles(wrap_to=st.just((u.Q(-180, "deg"), u.Q(180, "deg")))))
def test_symmetric_wrapping(angle):
    """Symmetric wrapping bounds work correctly."""
    assert angle.wrap_to is not None
    min_bound, max_bound = angle.wrap_to
    assert jnp.abs(min_bound.value) == jnp.abs(max_bound.value)
```

### Testing Angle Arrays

```python
@given(angles=cxst.angles(shape=10))
def test_angle_vector_operations(angles):
    """Angle vectors support common operations."""
    assert angles.shape == (10,)
    # Can convert units
    in_radians = angles.to("rad")
    assert in_radians.unit == u.unit("rad")


@given(angles=cxst.angles(shape=(5, 3), units="rad"))
def test_2d_angle_arrays(angles):
    """2D angle arrays work as expected."""
    assert angles.shape == (5, 3)
    assert angles.unit == u.unit("rad")
```

## Advanced Patterns

### Combining Multiple Strategies

```python
@given(
    angle1=cxst.angles(units="deg"),
    angle2=cxst.angles(units="deg"),
)
def test_angle_arithmetic(angle1, angle2):
    """Angles support arithmetic operations."""
    # Can add/subtract angles
    diff = angle1 - angle2
    assert isinstance(diff, u.Quantity)
```

### Using Assumptions

```python
@given(angle=cxst.angles(units="deg", min_value=-180, max_value=180))
def test_angle_in_range(angle):
    """Angles are within specified range."""
    assume(angle.value != 0)  # Skip zero if needed
    assert -180 <= angle.to("deg").value <= 180
```

### Dynamic Shapes

```python
@given(angle=cxst.angles(shape=st.tuples(st.integers(1, 10), st.integers(1, 10))))
def test_dynamic_shaped_angles(angle):
    """Angles with dynamically generated shapes."""
    assert len(angle.shape) == 2
    assert 1 <= angle.shape[0] <= 10
    assert 1 <= angle.shape[1] <= 10
```

## Testing Coordinate Transformations

## Integration with `unxt-hypothesis`

The {mod}`coordinax.hypothesis` package builds on top of `unxt-hypothesis` strategies. You can use both packages together:

```python
from hypothesis import given
import unxt_hypothesis as ust
import coordinax.hypothesis.main as cxst
import unxt as u


@given(
    angle=cxst.angles(units="rad"),
    distance=ust.quantities(units="kpc"),
)
def test_angle_and_distance(angle, distance):
    """Test using both angle and distance quantities."""
    assert isinstance(angle, u.Angle)
    assert isinstance(distance, u.Q)
    # Convert angle to degrees
    angle_deg = angle.to("deg")
    # Convert distance to parsecs
    distance_pc = distance.to("pc")
```

## Best Practices

### Use Specific Units When Needed

```python
# Good: Specific units prevent unit mismatches
@given(angle=cxst.angles(units="deg"))
def test_with_specific_units(angle):
    assert angle.unit == u.unit("deg")


# Less specific: Units may vary
@given(angle=cxst.angles())
def test_with_any_units(angle):
    # angle.unit could be any angle unit
    pass
```

### Test Edge Cases with `@example`

Use Hypothesis's `@example` decorator to combine property-based testing with specific edge cases. This ensures known edge cases are always tested while also running the full property-based test suite:

```python
from hypothesis import given, example
import coordinax.main as cx


@given(angle=cxst.angles(units="deg"))
@example(angle=cx.Angle(0, "deg"))  # Zero angle
@example(angle=cx.Angle(360, "deg"))  # Full circle
@example(angle=cx.Angle(-180, "deg"))  # Negative angle
def test_angle_properties_with_edge_cases(angle):
    """Test angle properties with both random and specific values."""
    assert isinstance(angle, cx.Angle)
    # Your property tests here
    angle_rad = angle.to("rad")
    assert isinstance(angle_rad, cx.Angle)


@given(distance=cxst.distances(units="kpc"))
@example(distance=cx.Distance(0, "kpc"))  # Zero distance
@example(distance=cx.Distance(1, "kpc"))  # Unit distance
def test_distance_properties_with_edge_cases(distance):
    """Test distance properties including edge cases."""
    assert isinstance(distance, cx.Distance)
    assert distance.value >= 0
```

The `@example` decorator runs before the property-based examples, ensuring your edge cases are always tested even if Hypothesis doesn't generate them randomly.

## Using `st.from_type()` with Distance Types

The `coordinax.hypothesis` package automatically registers strategies for core distance types with Hypothesis's `st.from_type()` function. This allows you to use these types in function annotations and let Hypothesis automatically generate test values.

### Registered Types

The following core coordinax type works with `st.from_type()`:

- `coordinax.distances.Distance`

For astro-specific distance types (for example `DistanceModulus` and `Parallax`), use the `coordinax.astro` package and its hypothesis strategies.

### Basic Usage

```python
from hypothesis import given, strategies as st
import coordinax.main as cx


# Hypothesis automatically knows how to generate these types
@given(dist=st.from_type(cx.Distance))
def test_distance_conversion(dist):
    """Test that distances can be converted between units."""
    assert isinstance(dist, cx.Distance)
    # Convert to different units
    dist_kpc = dist.to("kpc")
    assert dist_kpc.unit == "kpc"
```

### With `st.builds()`

The `st.from_type()` integration works seamlessly with `st.builds()` for testing functions that take distance types as arguments:

```python
from hypothesis import given, strategies as st
import coordinax.distances as cxd


def compute_absolute_magnitude(dist: cxd.Distance, apparent_mag: float) -> float:
    """Compute absolute magnitude from distance and apparent magnitude."""
    dm = dist.distance_modulus
    return apparent_mag - dm.value.item()


# Hypothesis automatically generates Distance instances
@given(
    result=st.builds(
        compute_absolute_magnitude,
        dist=st.from_type(cxd.Distance),
        apparent_mag=st.floats(min_value=-5, max_value=25),
    )
)
def test_absolute_magnitude(result):
    """Test absolute magnitude calculation."""
    assert isinstance(result, float)
    assert -30 < result < 30  # Reasonable range
```

### Combining with Other Strategies

You can combine `from_type()` with other Hypothesis features:

```python
@given(
    distances=st.lists(st.from_type(cxd.Distance), min_size=1, max_size=10),
)
def test_distance_statistics(distances):
    """Test statistical properties of distance collections."""
    import jax.numpy as jnp

    values = jnp.array([d.value.item() for d in distances])
    assert jnp.all(values >= 0)
    assert len(values) == len(distances)


@given(data=st.data())
def test_interactive_generation(data):
    """Test using data.draw() with from_type()."""
    # Generate a core distance type
    dist = data.draw(st.from_type(cxd.Distance))

    assert isinstance(dist, cxd.Distance)
```

## Debugging Failed Tests

When Hypothesis finds a failing test case, it will try to "shrink" the input to find the minimal failing example:

```python
@given(angle=cxst.angles(units="deg"))
def test_something(angle):
    # If this fails, Hypothesis will try to find the simplest failing angle
    assert some_property(angle)
```

You can also use `@reproduce_failure` to re-run a specific failing case:

```python
from hypothesis import reproduce_failure


@reproduce_failure("6.72.0", b"...")  # Hypothesis provides this
@given(angle=cxst.angles())
def test_something(angle):
    pass
```

## Performance Tips

1. Use `--hypothesis-profile` to control test settings
2. Use `assume()` to filter inputs early
3. Use specific constraints (min_value, max_value) instead of `assume()`
4. Keep property checks simple and fast

## Contributing

Found a bug or want to add more strategies? Contributions are welcome! Please see the [coordinax contributing guide](https://github.com/GalacticDynamics/coordinax/blob/main/CONTRIBUTING.md) for details.
