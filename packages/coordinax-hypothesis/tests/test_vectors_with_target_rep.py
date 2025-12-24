"""Tests for the vectors_with_target_rep strategy."""

import jax.numpy as jnp
from coordinax_hypothesis import representations, vectors_with_target_rep
from hypothesis import given, settings

import coordinax as cx


@given(
    vec_and_chain=vectors_with_target_rep(
        rep=representations(filter=cx.r.Abstract3D),
        role=cx.r.Pos,
    )
)
@settings(max_examples=30)
def test_position_vector_with_chain(vec_and_chain: tuple) -> None:
    """Test position vectors return single-element target chain."""
    vec, target_chain = vec_and_chain

    assert isinstance(vec, cx.vecs.Vector)
    assert isinstance(vec.role, cx.r.Pos)

    assert len(target_chain) == 1
    (target_rep,) = target_chain
    assert isinstance(target_rep, cx.r.Abstract3D)

    assert vec.rep.dimensionality == target_rep.dimensionality


@given(
    vec_and_chain=vectors_with_target_rep(
        rep=representations(filter=cx.r.Abstract3D),
        role=cx.r.Vel,
    )
)
@settings(max_examples=30)
def test_velocity_vector_with_chain(vec_and_chain: tuple) -> None:
    """Test velocity vectors return two-element target chain."""
    vec, target_chain = vec_and_chain

    assert isinstance(vec, cx.vecs.Vector)
    assert isinstance(vec.role, cx.r.Vel)

    assert len(target_chain) == 2
    vel_target, pos_target = target_chain
    assert isinstance(vel_target, cx.r.Abstract3D)
    assert isinstance(pos_target, cx.r.Abstract3D)

    assert vec.rep.dimensionality == vel_target.dimensionality
    assert vec.rep.dimensionality == pos_target.dimensionality


@given(
    vec_and_chain=vectors_with_target_rep(
        rep=representations(filter=cx.r.Abstract3D),
        role=cx.r.Acc,
    )
)
@settings(max_examples=30)
def test_acceleration_vector_with_chain(vec_and_chain: tuple) -> None:
    """Test acceleration vectors return three-element target chain."""
    vec, target_chain = vec_and_chain

    assert isinstance(vec, cx.vecs.Vector)
    assert isinstance(vec.role, cx.r.Acc)

    assert len(target_chain) == 3
    acc_target, vel_target, pos_target = target_chain
    assert isinstance(acc_target, cx.r.Abstract3D)
    assert isinstance(vel_target, cx.r.Abstract3D)
    assert isinstance(pos_target, cx.r.Abstract3D)

    assert vec.rep.dimensionality == acc_target.dimensionality
    assert vec.rep.dimensionality == vel_target.dimensionality
    assert vec.rep.dimensionality == pos_target.dimensionality


@given(
    vec_and_chain=vectors_with_target_rep(
        rep=cx.r.cart3d,
        role=cx.r.Pos,
    )
)
@settings(max_examples=20)
def test_specific_3d_position_vector(vec_and_chain: tuple) -> None:
    """Test vectors with specific 3D position representation."""
    vec, target_chain = vec_and_chain

    assert vec.rep == cx.r.cart3d

    (target_rep,) = target_chain
    assert isinstance(target_rep, cx.r.Abstract3D)
    assert vec.rep.dimensionality == target_rep.dimensionality


@given(
    vec_and_chain=vectors_with_target_rep(
        rep=cx.r.polar2d,
        role=cx.r.Vel,
    )
)
@settings(max_examples=20)
def test_specific_2d_velocity_vector(vec_and_chain: tuple) -> None:
    """Test vectors with specific 2D velocity representation."""
    vec, target_chain = vec_and_chain

    assert vec.rep == cx.r.polar2d
    assert isinstance(vec.role, cx.r.Vel)

    vel_target, pos_target = target_chain
    assert isinstance(vel_target, cx.r.Abstract2D)
    assert isinstance(pos_target, cx.r.Abstract2D)


@given(
    vec_and_chain=vectors_with_target_rep(
        rep=cx.r.radial1d,
        role=cx.r.Acc,
    )
)
@settings(max_examples=20)
def test_specific_1d_acceleration_vector(vec_and_chain: tuple) -> None:
    """Test vectors with specific 1D acceleration representation."""
    vec, target_chain = vec_and_chain

    assert vec.rep == cx.r.radial1d
    assert isinstance(vec.role, cx.r.Acc)

    assert len(target_chain) == 3
    for target_rep in target_chain:
        assert isinstance(target_rep, cx.r.Abstract1D)


@given(
    vec_and_chain=vectors_with_target_rep(
        rep=representations(filter=cx.r.Abstract3D),
        role=cx.r.Pos,
        shape=(5,),
    )
)
@settings(max_examples=20, deadline=None)  # flaky timing
def test_batched_position_vectors(vec_and_chain: tuple) -> None:
    """Test batched vectors preserve shape."""
    vec, target_chain = vec_and_chain

    assert vec.shape == (5,)

    (target_rep,) = target_chain
    converted = vec.vconvert(target_rep)
    assert converted.shape == (5,)


@given(
    vec_and_chain=vectors_with_target_rep(
        rep=representations(filter=cx.r.Abstract3D),
        role=cx.r.Vel,
        shape=(3, 4),
    )
)
@settings(max_examples=20)
def test_batched_3d_velocity_vectors(vec_and_chain: tuple) -> None:
    """Test batched 3D velocity vectors."""
    vec, target_chain = vec_and_chain

    assert vec.shape == (3, 4)
    assert isinstance(vec.role, cx.r.Vel)
    assert isinstance(vec.rep, cx.r.Abstract3D)

    vel_target, pos_target = target_chain
    assert isinstance(vel_target, cx.r.Abstract3D)
    assert isinstance(pos_target, cx.r.Abstract3D)


@given(
    vec_and_chain=vectors_with_target_rep(
        rep=representations(filter=cx.r.Abstract3D),
        role=cx.r.Pos,
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
    vec_and_chain=vectors_with_target_rep(
        rep=representations(filter=cx.r.Abstract2D),
        role=cx.r.Pos,
    )
)
@settings(max_examples=20, deadline=400)
def test_vector_conversion_to_target(vec_and_chain: tuple) -> None:
    """Test that position vectors can be converted to target representations."""
    vec, target_chain = vec_and_chain

    for target_rep in target_chain:
        converted = vec.vconvert(target_rep)
        assert converted.rep == target_rep
        assert isinstance(converted, cx.vecs.Vector)


@given(
    vec_and_chain=vectors_with_target_rep(
        rep=representations(filter=cx.r.Abstract2D),
        role=cx.r.Pos,
    )
)
@settings(max_examples=20)
def test_position_with_any_target(vec_and_chain: tuple) -> None:
    """Test position vector with unrestricted target."""
    vec, target_chain = vec_and_chain

    assert isinstance(vec.role, cx.r.Pos)
    assert isinstance(target_chain, tuple)
    assert len(target_chain) >= 1

    for target_rep in target_chain:
        assert target_rep.dimensionality == vec.rep.dimensionality


@given(
    vec_and_chain=vectors_with_target_rep(
        rep=cx.r.radial1d,
        role=cx.r.Acc,
        shape=(),
    )
)
@settings(max_examples=20)
def test_scalar_acceleration_vector(vec_and_chain: tuple) -> None:
    """Test scalar (non-batched) acceleration vectors."""
    vec, target_chain = vec_and_chain

    assert vec.shape == ()
    assert len(target_chain) == 3
    for target_rep in target_chain:
        assert isinstance(target_rep, cx.r.Abstract1D)
