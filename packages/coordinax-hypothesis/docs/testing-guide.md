# Testing with `coordinax-hypothesis`

This guide shows how to use the `coordinax-hypothesis` package for
property-based testing of code that uses coordinax angles.

## What is Property-Based Testing?

Property-based testing is a testing methodology where you specify properties
that should hold true for all inputs, and the testing framework (Hypothesis)
generates random test cases to verify those properties.

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
import coordinax_hypothesis as cxst


@given(angle=cxst.angles(units="deg"))
def test_angle_properties(angle):
    """All angles have valid values and units."""
    assert isinstance(angle, u.Angle)
    assert angle.value is not None
    assert angle.unit is not None
```

Hypothesis will generate random test cases with different values, uncovering
edge cases you might not have thought of.

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

## Basic Examples

```python
from hypothesis import given, assume, strategies as st
import coordinax_hypothesis as cxst
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
    assert isinstance(diff, u.Quantity)  # Result is a Quantity
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

## Integration with unxt-hypothesis

The `coordinax-hypothesis` package builds on top of
[unxt-hypothesis](https://github.com/GalacticDynamics/unxt) strategies. You can
use both packages together:

```python
from hypothesis import given
import unxt_hypothesis as ust
import coordinax_hypothesis as cxst
import unxt as u


@given(
    angle=cxst.angles(units="rad"),
    distance=ust.quantities(units="kpc"),
)
def test_angle_and_distance(angle, distance):
    """Test using both angle and distance quantities."""
    assert isinstance(angle, u.Angle)
    assert isinstance(distance, u.Quantity)
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

### Control Example Count

```python
from hypothesis import settings


@given(angle=cxst.angles())
@settings(max_examples=100)  # Run 100 test cases
def test_with_more_examples(angle):
    assert isinstance(angle, u.Angle)
```

### Test Edge Cases with `@example`

Use Hypothesis's `@example` decorator to combine property-based testing with
specific edge cases. This ensures known edge cases are always tested while also
running the full property-based test suite:

```python
from hypothesis import given, example
import coordinax as cx


@given(angle=cxst.angles(units="deg"))
@example(angle=u.Angle(0, "deg"))  # Zero angle
@example(angle=u.Angle(360, "deg"))  # Full circle
@example(angle=u.Angle(-180, "deg"))  # Negative angle
def test_angle_properties_with_edge_cases(angle):
    """Test angle properties with both random and specific values."""
    assert isinstance(angle, u.Angle)
    # Your property tests here
    angle_rad = angle.to("rad")
    assert isinstance(angle_rad, u.Angle)


@given(distance=cxst.distances(units="kpc"))
@example(distance=cx.Distance(0, "kpc"))  # Zero distance
@example(distance=cx.Distance(1, "kpc"))  # Unit distance
def test_distance_properties_with_edge_cases(distance):
    """Test distance properties including edge cases."""
    assert isinstance(distance, cx.Distance)
    assert distance.value >= 0
```

The `@example` decorator runs before the property-based examples, ensuring your
edge cases are always tested even if Hypothesis doesn't generate them randomly.

## Debugging Failed Tests

When Hypothesis finds a failing test case, it will try to "shrink" the input to
find the minimal failing example:

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

1. Use `settings(max_examples=N)` to control test duration
2. Use `assume()` to filter inputs early
3. Use specific constraints (min_value, max_value) instead of `assume()`
4. Keep property checks simple and fast

## Contributing

Found a bug or want to add more strategies? Contributions are welcome! Please
see the
[coordinax contributing guide](https://github.com/GalacticDynamics/coordinax/blob/main/CONTRIBUTING.md)
for details.
