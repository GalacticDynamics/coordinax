"""Test vectors."""

import coordinax_hypothesis as cxst
import jax.tree as jtu
from hypothesis import given, settings

import quaxed.numpy as jnp

import coordinax as cx


@given(
    vec_and_rep=cxst.vectors_with_target_rep(rep=cxst.representations(), role=cx.r.Pos)
)
def test_position_vector_vconvert_roundtrip(vec_and_rep):
    """Test vconvert roundtrip for position vectors."""
    vec, target_chain = vec_and_rep

    for rep in target_chain:
        converted = vec.vconvert(rep)
        assert isinstance(converted, cx.Vector)
        assert converted.rep == rep

        roundtripped = converted.vconvert(vec.rep)
        assert isinstance(roundtripped, cx.Vector)
        assert roundtripped.rep == vec.rep
        assert roundtripped.role == vec.role
        assert all(jtu.map(jnp.allclose, vec, roundtripped))


@settings(max_examples=10, deadline=None)
@given(
    vec_and_rep=cxst.vectors_with_target_rep(rep=cxst.representations(), role=cx.r.Pos)
)
def test_position_represent_as_matches_vconvert(vec_and_rep):
    """Test that represent_as matches vconvert for position vectors."""
    vec, target_chain = vec_and_rep

    for rep in target_chain:
        # Test that represent_as produces same result as vconvert
        result_represent_as = vec.represent_as(rep)
        result_vconvert = vec.vconvert(rep)

        assert isinstance(result_represent_as, cx.Vector)
        assert isinstance(result_vconvert, cx.Vector)
        assert result_represent_as.rep == result_vconvert.rep
        assert result_represent_as.role == result_vconvert.role
        # Compare data values only, not the entire pytree
        assert all(
            jtu.map(
                jnp.allclose,
                list(result_represent_as.data.values()),
                list(result_vconvert.data.values()),
            )
        )


@settings(max_examples=10, deadline=None)
@given(
    vec_and_rep=cxst.vectors_with_target_rep(rep=cxst.representations(), role=cx.r.Vel)
)
def test_velocity_represent_as_matches_vconvert(vec_and_rep):
    """Test that represent_as matches vconvert for velocity vectors."""
    vec, target_chain = vec_and_rep

    # Create a position vector for the differential transformation
    pos = cx.Vector.from_([1.0, 2.0, 3.0], "m")
    pos_in_vec_rep = pos.vconvert(vec.rep)

    for rep in target_chain:
        # Test that represent_as produces same result as vconvert
        result_represent_as = vec.represent_as(rep, pos_in_vec_rep)
        result_vconvert = vec.vconvert(rep, pos_in_vec_rep)

        assert isinstance(result_represent_as, cx.Vector)
        assert isinstance(result_vconvert, cx.Vector)
        assert result_represent_as.rep == result_vconvert.rep
        assert result_represent_as.role == result_vconvert.role
        # Compare data values only, not the entire pytree
        assert all(
            jtu.map(
                jnp.allclose,
                list(result_represent_as.data.values()),
                list(result_vconvert.data.values()),
            )
        )


@settings(max_examples=10, deadline=None)
@given(
    vec_and_rep=cxst.vectors_with_target_rep(rep=cxst.representations(), role=cx.r.Acc)
)
def test_acceleration_represent_as_matches_vconvert(vec_and_rep):
    """Test that represent_as matches vconvert for acceleration vectors."""
    vec, target_chain = vec_and_rep

    # Create a position vector for the differential transformation
    pos = cx.Vector.from_([1.0, 2.0, 3.0], "m")
    pos_in_vec_rep = pos.vconvert(vec.rep)

    for rep in target_chain:
        # Test that represent_as produces same result as vconvert
        result_represent_as = vec.represent_as(rep, pos_in_vec_rep)
        result_vconvert = cx.vconvert(rep, vec, pos_in_vec_rep)

        assert isinstance(result_represent_as, cx.Vector)
        assert isinstance(result_vconvert, cx.Vector)
        assert result_represent_as.rep == result_vconvert.rep
        assert result_represent_as.role == result_vconvert.role
        assert all(
            jtu.map(
                jnp.allclose,
                list(result_represent_as.data.values()),
                list(result_vconvert.data.values()),
            )
        )
