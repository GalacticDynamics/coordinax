# Testing with `coordinax-hypothesis`

This guide shows how to use the `coordinax-hypothesis` package for
property-based testing of code that uses coordinax angles.

## What is Property-Based Testing?

Property-based testing is a testing methodology where you specify properties
that should hold true for all inputs, and the testing framework (Hypothesis)
generates random test cases to verify those properties.

Instead of writing:

```
import unxt as u


def test_angle_normalization():
    angle = u.Angle(370, "deg")
    assert angle.to("deg").value == 370
```

You write:

```
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

```
from hypothesis import given, assume, strategies as st
import coordinax_hypothesis as cxst
import unxt as u
import jax.numpy as jnp
```

### Testing Angle Properties

```
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

```
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

```
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

```
@given(
    angle1=cxst.angles(units="deg"),
    angle2=cxst.angles(units="deg"),
)
def test_angle_arithmetic(angle1, angle2):
    """Angles support arithmetic operations."""
    # Can add/subtract angles
    diff = angle1 - angle2
    assert isinstance(diff, u.Qsult is a Quantity
```

### Using Assumptions

```
@given(angle=cxst.angles(units="deg", min_value=-180, max_value=180))
def test_angle_in_range(angle):
    """Angles are within specified range."""
    assume(angle.value != 0)  # Skip zero if needed
    assert -180 <= angle.to("deg").value <= 180
```

### Dynamic Shapes

```
@given(angle=cxst.angles(shape=st.tuples(st.integers(1, 10), st.integers(1, 10))))
def test_dynamic_shaped_angles(angle):
    """Angles with dynamically generated shapes."""
    assert len(angle.shape) == 2
    assert 1 <= angle.shape[0] <= 10
    assert 1 <= angle.shape[1] <= 10
```

## Testing Representation Classes

### Basic Representation Class Testing

```
from hypothesis import given
import coordinax as cx
import coordinax_hypothesis as cxst


@given(chart_class=cxst.chart_classes())
def test_any_chart_class(chart_class):
    """Any chart class can be tested."""
    assert issubclass(chart_class, cx.charts.AbstractChart)
```

### Testing Multiple Representation Types

```
@given(
    chart_class=cxst.chart_classes(
        filter=(cx.charts.Abstract3D, cx.charts.AbstractSpherical3D)
    )
)
def test_spherical_3d_chartresentations(chart_class):
    """Test charts that are spherical 3D."""
    assert issubclass(chart_class, (cx.charts.Abstract3D, cx.charts.AbstractSpherical3D))
```

### Dynamically Choosing Representation Types

```
from hypothesis import strategies as st


@given(
    chart_class=cxst.chart_classes(
        filter=st.sampled_from(
            [
                cx.charts.Abstract1D,
                cx.charts.Abstract2D,
                cx.charts.Abstract3D,
            ]
        )
    )
)
def test_random_chart_type(chart_class):
    """Test with randomly chosen chart category."""
    assert issubclass(chart_class, (cx.charts.Abstract1D, cx.charts.Abstract2D, cx.charts.Abstract3D))
```

### Testing Chart Construction with `chart_init_kwargs`

The `chart_init_kwargs` strategy generates valid initialization arguments for
chart classes. This is useful when you want to test chart construction or need
to create charts dynamically with varying parameters:

```
from hypothesis import given
import coordinax.charts as cxc
import coordinax_hypothesis as cxst


# Generate valid kwargs for specific chart classes
@given(kwargs=cxst.chart_init_kwargs(cxc.SpaceTimeCT))
def test_spacetime_construction(kwargs):
    """Test SpaceTimeCT construction with generated kwargs."""
    # kwargs will contain 'spatial_chart' and 'c'
    assert 'spatial_chart' in kwargs
    chart = cxc.SpaceTimeCT(**kwargs)
    assert isinstance(chart, cxc.SpaceTimeCT)
    assert chart.ndim == kwargs['spatial_chart'].ndim + 1


# Test embedded manifold construction
@given(kwargs=cxst.chart_init_kwargs(cxc.EmbeddedManifold))
def test_embedded_manifold_construction(kwargs):
    """Test EmbeddedManifold construction with generated kwargs."""
    assert 'intrinsic_chart' in kwargs
    assert 'ambient_chart' in kwargs
    assert 'params' in kwargs
    chart = cxc.EmbeddedManifold(**kwargs)
    assert isinstance(chart, cxc.EmbeddedManifold)


# Combine with chart_classes for generic testing
@given(chart_cls=st.sampled_from([cxc.Cart1D, cxc.Polar2D, cxc.Spherical3D]))
def test_various_chart_construction(chart_cls):
    """Test construction of various chart classes."""
    kwargs = cxst.chart_init_kwargs(chart_cls).example()
    chart = chart_cls(**kwargs)
    assert isinstance(chart, chart_cls)


# Test that kwargs can be used with different instances
@given(kwargs=cxst.chart_init_kwargs(cxc.Cylindrical3D))
def test_kwargs_reusable(kwargs):
    """Test that generated kwargs can create multiple instances."""
    chart1 = cxc.Cylindrical3D(**kwargs)
    chart2 = cxc.Cylindrical3D(**kwargs)
    # Both should be valid instances
    assert isinstance(chart1, cxc.Cylindrical3D)
    assert isinstance(chart2, cxc.Cylindrical3D)
    assert chart1.ndim == chart2.ndim == 3
```

## Testing Coordinate Transformations

### Testing with `charts_like`

The `charts_like` strategy generates charts that match the flags of a template,
making it easy to test transformations across compatible charts:

```
from hypothesis import given
import coordinax as cx
import coordinax_hypothesis as cxst


# Test that 3D charts can be converted to each other
@given(
    source_chart=t=cxst.charts(filter=cx.charts.Abstract3D),
    target_chart=t=cxst.charts_like(cxst.charts(filter=cx.charts.Abstract3D)),
)
def test_3d_chart_conversions(source_chart, target_chart):
    """Test conversions between 3D charts."""
    # Both charts are 3D
    assert isinstance(source_chart, cx.charts.Abstract3D)
    assert isinstance(target_chart, cx.charts.Abstract3D)
    assert source_chart.ndim == target_chart.ndim == 3


# Test 2D chart transformations
@given(chart=t=cxst.charts_like(cx.charts.polar2d))
def test_charts_like_polar(chart):
    """Generate charts with same flags as Polar2D."""
    assert isinstance(chart, cx.charts.Abstract2D)
    # Could be Cart2D, Polar2D, TwoSphere, etc.
```

### Testing with `chart_time_chain`

The `chart_time_chain` strategy generates chains of time antiderivatives given a
role flag and a chart, useful for testing conversions across acceleration →
velocity → position:

```
# Test that acceleration charts have valid time derivative chains
@given(chain=cxst.chart_time_chain(cx.roles.PhysAcc, cx.charts.cart3d))
def test_acceleration_chain(chain):
    """Test the full time derivative chain from acceleration."""
    acc_chart, vel_chart, point_chart = chain

    # Verify types
    # All maintain dimensionality
    assert isinstance(acc_chart, cx.charts.Abstract3D)
    assert isinstance(vel_chart, cx.charts.Abstract3D)
    assert isinstance(point_chart, cx.charts.Abstract3D)


# Test velocity chains
@given(chain=cxst.chart_time_chain(cx.roles.PhysVel, cx.charts.polar2d))
def test_velocity_chain(chain):
    """Test time derivative chain from velocity."""
    vel_chart, point_chart = chain

    assert isinstance(vel_chart, cx.charts.Abstract2D)
    assert isinstance(point_chart, cx.charts.Abstract2D)
    assert vel_chart.ndim == point_chart.ndim

# Test that position charts return single-element chains
@given(chain=cxst.chart_time_chain(cx.roles.PhysDisp, cx.charts.sph3d))
def test_position_chain_is_singleton(chain):
    """Position charts have no time antiderivative."""
    assert len(chain) == 1
    (point_chart,) = chain
    assert isinstance(point_chart, cx.charts.Abstract3D)
```

### Testing with `vectors_with_target_chart`

The `vectors_with_target_chart` strategy generates a vector and a full chain of
compatible target charts, perfect for testing conversions. The target chain
automatically matches the flags of the source vector:

```
# Test position vector conversions
@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=t=cxst.charts(filter=cx.charts.Abstract3D),
        role=cx.roles.PhysDisp,
    )
)
def test_position_vector_conversions(vec_and_chain):
    """Test that position vectors can convert to compatible charts."""
    vec, (target_chart,) = vec_and_chain

    # Verify source and target are compatible
    assert isinstance(vec.role, cx.roles.PhysDisp)
    assert isinstance(vec.chart, cx.charts.Abstract3D)
    assert isinstance(target_chart, cx.charts.Abstract3D)
    assert vec.chart.ndim == target_chart.ndim

    # Test conversion
    converted = vec.vconvert(target_chart)
    assert converted.chart == target_chart


# Test velocity vector conversions with full chain
@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=t=cx.charts.cart3d,
        role=cx.roles.PhysVel,
    ),
    point_vec=cxst.vectors(chart=t=cx.charts.cart3d, role=cx.roles.PhysDisp),
)
def test_velocity_vector_conversion_chain(vec_and_chain, point_vec):
    """Test velocity vectors can convert to velocity and position reps."""
    vec, target_chain = vec_and_chain

    # Chain is (vel_chart, point_chart)
    assert len(target_chain) == 2
    vel_target, point_target = target_chain

    # Test conversion to each target
    vel_converted = vec.vconvert(vel_target, point_vec)
    assert vel_converted.chart == vel_target

    point_converted = vec.vconvert(point_target, point_vec)
    assert point_converted.chart == point_target


# Test acceleration vectors with full chain
@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=t=cx.charts.cart3d,
        role=cx.roles.PhysAcc,
        shape=(5,),  # Test with batched vectors
    ),
    point_vec=cxst.vectors(chart=t=cx.charts.cart3d, role=cx.roles.PhysDisp, shape=(5,)),
)
def test_batched_acceleration_conversions(vec_and_chain, point_vec):
    """Test batched acceleration vector conversions."""
    vec, target_chain = vec_and_chain

    # Verify shape is preserved
    assert vec.shape == (5,)

    # Chain is (acc_chart, vel_chart, point_chart)
    assert len(target_chain) == 3

    # Test all conversions preserve shape
    for target_chart in target_chain:
        converted = vec.vconvert(target_chart, point_vec)
        assert converted.shape == (5,)
        assert converted.chart == target_chart
```

### Testing Conversion Properties

Use these strategies to test mathematical properties of coordinate
transformations:

```
# Test that conversions are invertible
@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=t=cx.charts.cart3d,
        role=cx.roles.PhysDisp,
    )
)
def test_conversion_roundtrip(vec_and_chain):
    """Test that converting and converting back preserves the vector."""
    vec, (target_chart,) = vec_and_chain

    # Convert to target and back
    converted = vec.vconvert(target_chart)
    back = converted.vconvert(vec.chart)

    # Should be close to original (within numerical precision)
    import jax.numpy as jnp

    for key in vec.data.keys():
        assert jnp.allclose(vec.data[key].value, back.data[key].value, rtol=1e-5)


# Test that conversions preserve norms for position vectors
@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=t=cxst.charts(filter=cx.charts.Abstract3D),
        role=cx.roles.PhysDisp,
    )
)
def test_conversion_preserves_norm(vec_and_chain):
    """Test that coordinate transformations preserve vector norms."""
    vec, (target_chart,) = vec_and_chain

    # Get norms (requires both to support norm calculation)
    try:
        original_norm = vec.norm()
        converted = vec.vconvert(target_chart)
        converted_norm = converted.norm()

        # Norms should be identical
        import jax.numpy as jnp

        assert jnp.allclose(original_norm.value, converted_norm.value, rtol=1e-6)
    except (AttributeError, NotImplementedError):
        # Not all charts support norm calculation
        pass
```

## Integration with `unxt-hypothesis`

The {mod}`coordinax-hypothesis` package builds on top of {mod}`unxt-hypothesis`
strategies. You can use both packages together:

```
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
    assert isinstance(distance, u.Q)
    # Convert angle to degrees
    angle_deg = angle.to("deg")
    # Convert distance to parsecs
    distance_pc = distance.to("pc")
```

## Best Practices

### Use Specific Units When Needed

```
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

```
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

```
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

## Using `st.from_type()` with Distance Types

The `coordinax-hypothesis` package automatically registers strategies for
distance types with Hypothesis's `st.from_type()` function. This allows you to
use these types in function annotations and let Hypothesis automatically
generate test values.

### Registered Types

The following coordinax types work with `st.from_type()`:

- `coordinax.Distance`
- `coordinax.DistanceModulus`
- `coordinax.Parallax`

### Basic Usage

```
from hypothesis import given, strategies as st
import coordinax.distances as cxd


# Hypothesis automatically knows how to generate these types
@given(dist=st.from_type(cxd.Distance))
def test_distance_conversion(dist):
    """Test that distances can be converted between units."""
    assert isinstance(dist, cxd.Distance)
    # Convert to different units
    dist_kpc = dist.to("kpc")
    assert dist_kpc.unit == "kpc"


@given(dm=st.from_type(cxd.DistanceModulus))
def test_distance_modulus_properties(dm):
    """Test distance modulus properties."""
    assert isinstance(dm, cxd.DistanceModulus)
    assert dm.unit == "mag"  # Always in magnitudes


@given(plx=st.from_type(cxd.Parallax))
def test_parallax_to_distance(plx):
    """Test converting parallax to distance."""
    assert isinstance(plx, cxd.Parallax)
    # Can convert to distance
    dist = cxd.Distance.from_(1 / plx.value, "pc")
    assert isinstance(dist, cxd.Distance)
```

### With `st.builds()`

The `st.from_type()` integration works seamlessly with `st.builds()` for testing
functions that take distance types as arguments:

```
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


def distance_from_parallax(plx: cxd.Parallax) -> cxd.Distance:
    """Convert parallax to distance."""
    return cxd.Distance.from_(1 / plx.value, "pc")


@given(result=st.builds(distance_from_parallax, plx=st.from_type(cxd.Parallax)))
def test_parallax_conversion(result):
    """Test parallax to distance conversion."""
    assert isinstance(result, cxd.Distance)
    assert result.value > 0
```

### Combining with Other Strategies

You can combine `from_type()` with other Hypothesis features:

```
@given(
    distances=st.lists(st.from_type(cxd.Distance), min_size=1, max_size=10),
)
def test_distance_statistics(distances):
    """Test statistical properties of distance collections."""
    import jax.numpy as jnp

    values = jnp.array([d.value.item() for d in distances])
    assert jnp.all(values >= 0)
    assert len(values) == len(distances)


@given(
    data=st.data(),
)
def test_interactive_generation(data):
    """Test using data.draw() with from_type()."""
    # Generate one of each type
    dist = data.draw(st.from_type(cxd.Distance))
    dm = data.draw(st.from_type(cxd.DistanceModulus))
    plx = data.draw(st.from_type(cxd.Parallax))

    assert isinstance(dist, cxd.Distance)
    assert isinstance(dm, cxd.DistanceModulus)
    assert isinstance(plx, cxd.Parallax)
```

## Debugging Failed Tests

When Hypothesis finds a failing test case, it will try to "shrink" the input to
find the minimal failing example:

```
@given(angle=cxst.angles(units="deg"))
def test_something(angle):
    # If this fails, Hypothesis will try to find the simplest failing angle
    assert some_property(angle)
```

You can also use `@reproduce_failure` to re-run a specific failing case:

```
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
