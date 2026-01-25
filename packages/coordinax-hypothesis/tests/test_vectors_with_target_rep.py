"""Tests for the vectors_with_target_chart strategy."""

import jax.numpy as jnp
from hypothesis import given, settings

import coordinax as cx
import coordinax_hypothesis as cxst


@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=cxst.charts(filter=cx.charts.Abstract3D),
        role=cx.roles.PhysDisp,
    )
)
@settings(max_examples=30)
def test_position_vector_with_chain(vec_and_chain: tuple) -> None:
    """Test position vectors return single-element target chain."""
    vec, target_chain = vec_and_chain

    assert isinstance(vec, cx.Vector)
    assert isinstance(vec.role, cx.roles.PhysDisp)

    assert len(target_chain) == 1
    (target_chart,) = target_chain
    assert isinstance(target_chart, cx.charts.Abstract3D)

    assert vec.chart.ndim == target_chart.ndim


@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=cxst.charts(filter=cx.charts.Abstract3D),
        role=cx.roles.PhysVel,
    )
)
@settings(max_examples=30)
def test_velocity_vector_with_chain(vec_and_chain: tuple) -> None:
    """Test velocity vectors return two-element target chain."""
    vec, target_chain = vec_and_chain

    assert isinstance(vec, cx.Vector)
    assert isinstance(vec.role, cx.roles.PhysVel)

    assert len(target_chain) == 2
    vel_target, pos_target = target_chain
    assert isinstance(vel_target, cx.charts.Abstract3D)
    assert isinstance(pos_target, cx.charts.Abstract3D)

    assert vec.chart.ndim == vel_target.ndim
    assert vec.chart.ndim == pos_target.ndim


@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=cxst.charts(filter=cx.charts.Abstract3D),
        role=cx.roles.PhysAcc,
    )
)
@settings(max_examples=30)
def test_acceleration_vector_with_chain(vec_and_chain: tuple) -> None:
    """Test acceleration vectors return three-element target chain."""
    vec, target_chain = vec_and_chain

    assert isinstance(vec, cx.Vector)
    assert isinstance(vec.role, cx.roles.PhysAcc)

    assert len(target_chain) == 3
    acc_target, vel_target, pos_target = target_chain
    assert isinstance(acc_target, cx.charts.Abstract3D)
    assert isinstance(vel_target, cx.charts.Abstract3D)
    assert isinstance(pos_target, cx.charts.Abstract3D)

    assert vec.chart.ndim == acc_target.ndim
    assert vec.chart.ndim == vel_target.ndim
    assert vec.chart.ndim == pos_target.ndim


@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=cx.charts.cart3d,
        role=cx.roles.PhysDisp,
    )
)
@settings(max_examples=20)
def test_specific_3d_position_vector(vec_and_chain: tuple) -> None:
    """Test vectors with specific 3D position representation."""
    vec, target_chain = vec_and_chain

    assert vec.chart == cx.charts.cart3d

    (target_chart,) = target_chain
    assert isinstance(target_chart, cx.charts.Abstract3D)
    assert vec.chart.ndim == target_chart.ndim


@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=cx.charts.polar2d,
        role=cx.roles.PhysVel,
    )
)
@settings(max_examples=20)
def test_specific_2d_velocity_vector(vec_and_chain: tuple) -> None:
    """Test vectors with specific 2D velocity representation."""
    vec, target_chain = vec_and_chain

    assert vec.chart == cx.charts.polar2d
    assert isinstance(vec.role, cx.roles.PhysVel)

    vel_target, pos_target = target_chain
    assert isinstance(vel_target, cx.charts.Abstract2D)
    assert isinstance(pos_target, cx.charts.Abstract2D)


@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=cx.charts.radial1d,
        role=cx.roles.PhysAcc,
    )
)
@settings(max_examples=20)
def test_specific_1d_acceleration_vector(vec_and_chain: tuple) -> None:
    """Test vectors with specific 1D acceleration representation."""
    vec, target_chain = vec_and_chain

    assert vec.chart == cx.charts.radial1d
    assert isinstance(vec.role, cx.roles.PhysAcc)

    assert len(target_chain) == 3
    for target_chart in target_chain:
        assert isinstance(target_chart, cx.charts.Abstract1D)


@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=cxst.charts(filter=cx.charts.Abstract3D),
        role=cx.roles.PhysDisp,
        shape=(5,),
    )
)
@settings(max_examples=20, deadline=None)  # flaky timing
def test_batched_position_vectors(vec_and_chain: tuple) -> None:
    """Test batched vectors preserve shape."""
    vec, target_chain = vec_and_chain

    assert vec.shape == (5,)

    (target_chart,) = target_chain
    converted = vec.vconvert(target_chart)
    assert converted.shape == (5,)


@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=cxst.charts(filter=cx.charts.Abstract3D),
        role=cx.roles.PhysVel,
        shape=(3, 4),
    )
)
@settings(max_examples=20)
def test_batched_3d_velocity_vectors(vec_and_chain: tuple) -> None:
    """Test batched 3D velocity vectors."""
    vec, target_chain = vec_and_chain

    assert vec.shape == (3, 4)
    assert isinstance(vec.role, cx.roles.PhysVel)
    assert isinstance(vec.chart, cx.charts.Abstract3D)

    vel_target, pos_target = target_chain
    assert isinstance(vel_target, cx.charts.Abstract3D)
    assert isinstance(pos_target, cx.charts.Abstract3D)


@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=cxst.charts(filter=cx.charts.Abstract3D),
        role=cx.roles.PhysDisp,
        dtype=jnp.float32,
    )
)
@settings(max_examples=20)
def test_float32_dtype(vec_and_chain: tuple) -> None:
    """Test vectors with float32 dtype."""
    vec, _ = vec_and_chain

    for component in vec.data.values():
        if hasattr(component, "dtype"):
            assert component.dtype == jnp.float32


@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=cxst.charts(filter=cx.charts.Abstract2D),
        role=cx.roles.PhysDisp,
    )
)
@settings(max_examples=20, deadline=400)
def test_vector_conversion_to_target(vec_and_chain: tuple) -> None:
    """Test that position vectors can be converted to target representations."""
    vec, target_chain = vec_and_chain

    for target_chart in target_chain:
        converted = vec.vconvert(target_chart)
        assert converted.chart == target_chart
        assert isinstance(converted, cx.Vector)


@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=cxst.charts(filter=cx.charts.Abstract2D),
        role=cx.roles.PhysDisp,
    )
)
@settings(max_examples=20)
def test_position_with_any_target(vec_and_chain: tuple) -> None:
    """Test position vector with unrestricted target."""
    vec, target_chain = vec_and_chain

    assert isinstance(vec.role, cx.roles.PhysDisp)
    assert isinstance(target_chain, tuple)
    assert len(target_chain) >= 1

    for target_chart in target_chain:
        assert target_chart.ndim == vec.chart.ndim


@given(
    vec_and_chain=cxst.vectors_with_target_chart(
        chart=cx.charts.radial1d,
        role=cx.roles.PhysAcc,
        shape=(),
    )
)
@settings(max_examples=20)
def test_scalar_acceleration_vector(vec_and_chain: tuple) -> None:
    """Test scalar (non-batched) acceleration vectors."""
    vec, target_chain = vec_and_chain

    assert vec.shape == ()
    assert len(target_chain) == 3
    for target_chart in target_chain:
        assert isinstance(target_chart, cx.charts.Abstract1D)
