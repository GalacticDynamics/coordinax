"""Unit tests for role derivative and antiderivative chains."""

from jaxtyping import TypeCheckError

import pytest
from hypothesis import given, settings

import coordinax.roles as cxr
import coordinax_hypothesis.core as cxst


class TestPhysicalDerivativeChain:
    """Test the physical role derivative chain: PhysDisp -> PhysVel -> PhysAcc."""

    def test_phys_disp_derivative_is_phys_vel(self) -> None:
        """PhysDisp.derivative() is PhysVel."""
        result = cxr.phys_disp.derivative()
        assert isinstance(result, cxr.PhysVel)

    def test_phys_vel_derivative_is_phys_acc(self) -> None:
        """PhysVel.derivative() is PhysAcc."""
        result = cxr.phys_vel.derivative()
        assert isinstance(result, cxr.PhysAcc)

    def test_phys_acc_derivative_is_not_implemented(self) -> None:
        """PhysAcc.derivative() raises (no 3rd-order role)."""
        with pytest.raises(TypeCheckError):
            cxr.phys_acc.derivative()


class TestPhysicalAntiderivativeChain:
    """Test the physical role antiderivative chain: PhysAcc -> PhysVel -> PhysDisp."""

    def test_phys_acc_antiderivative_is_phys_vel(self) -> None:
        """PhysAcc.antiderivative() is PhysVel."""
        result = cxr.phys_acc.antiderivative()
        assert isinstance(result, cxr.PhysVel)

    def test_phys_vel_antiderivative_is_phys_disp(self) -> None:
        """PhysVel.antiderivative() is PhysDisp."""
        result = cxr.phys_vel.antiderivative()
        assert isinstance(result, cxr.PhysDisp)

    def test_phys_disp_antiderivative_is_not_implemented(self) -> None:
        """PhysDisp.antiderivative() raises."""
        with pytest.raises(TypeCheckError):
            cxr.phys_disp.antiderivative()


class TestCoordDerivativeChain:
    """Test coordinate role derivative chain: CoordDisp -> CoordVel -> CoordAcc."""

    def test_coord_disp_derivative_is_coord_vel(self) -> None:
        """CoordDisp.derivative() is CoordVel."""
        result = cxr.coord_disp.derivative()
        assert isinstance(result, cxr.CoordVel)

    def test_coord_vel_derivative_is_coord_acc(self) -> None:
        """CoordVel.derivative() is CoordAcc."""
        result = cxr.coord_vel.derivative()
        assert isinstance(result, cxr.CoordAcc)

    def test_coord_acc_derivative_is_not_implemented(self) -> None:
        """CoordAcc.derivative() raises."""
        with pytest.raises(TypeCheckError):
            cxr.coord_acc.derivative()


class TestCoordAntiderivativeChain:
    """Test coordinate antiderivative chain: CoordAcc -> CoordVel -> CoordDisp."""

    def test_coord_acc_antiderivative_is_coord_vel(self) -> None:
        """CoordAcc.antiderivative() is CoordVel."""
        result = cxr.coord_acc.antiderivative()
        assert isinstance(result, cxr.CoordVel)

    def test_coord_vel_antiderivative_is_coord_disp(self) -> None:
        """CoordVel.antiderivative() is CoordDisp."""
        result = cxr.coord_vel.antiderivative()
        assert isinstance(result, cxr.CoordDisp)

    def test_coord_disp_antiderivative_is_not_implemented(self) -> None:
        """CoordDisp.antiderivative() raises."""
        with pytest.raises(TypeCheckError):
            cxr.coord_disp.antiderivative()


class TestPointDerivativeChain:
    """Test Point derivative/antiderivative behavior."""

    def test_point_derivative_is_phys_vel(self) -> None:
        """Point.derivative() is PhysVel (convenience: dp/dt has velocity units)."""
        result = cxr.point.derivative()
        assert isinstance(result, cxr.PhysVel)

    def test_point_antiderivative_is_not_implemented(self) -> None:
        """Point.antiderivative() raises (no position integral)."""
        with pytest.raises(TypeCheckError):
            cxr.point.antiderivative()


class TestDerivativeAntiderivativeInverse:
    """Test that derivative and antiderivative are inverse operations."""

    @pytest.mark.parametrize(
        "role",
        [cxr.phys_vel, cxr.coord_vel],
        ids=["PhysVel", "CoordVel"],
    )
    def test_derivative_antiderivative_roundtrip(self, role: cxr.AbstractRole) -> None:
        """derivative().antiderivative() returns the original role (order=1)."""
        derived = role.derivative()
        assert derived is not NotImplemented
        back = derived.antiderivative()
        assert type(back) is type(role)

    @pytest.mark.parametrize(
        "role",
        [cxr.phys_vel, cxr.coord_vel],
        ids=["PhysVel", "CoordVel"],
    )
    def test_antiderivative_derivative_roundtrip(self, role: cxr.AbstractRole) -> None:
        """antiderivative().derivative() returns the original role (order=1)."""
        integrated = role.antiderivative()
        assert integrated is not NotImplemented
        back = integrated.derivative()
        assert type(back) is type(role)

    @given(role=cxst.physical_roles())
    @settings(deadline=None)
    def test_phys_roles_chain_consistency(self, role: cxr.AbstractPhysRole) -> None:
        """For physical roles, derivative/antiderivative stay within PhysRole."""
        try:
            d = role.derivative()
        except TypeCheckError:
            pass  # Terminal role (e.g. PhysAcc has no derivative)
        else:
            assert isinstance(d, cxr.AbstractPhysRole)
        try:
            a = role.antiderivative()
        except TypeCheckError:
            pass  # Terminal role (e.g. PhysDisp has no antiderivative)
        else:
            assert isinstance(a, cxr.AbstractPhysRole)

    @pytest.mark.parametrize(
        "role",
        [cxr.coord_disp, cxr.coord_vel, cxr.coord_acc],
        ids=["CoordDisp", "CoordVel", "CoordAcc"],
    )
    def test_coord_roles_chain_consistency(self, role: cxr.AbstractCoordRole) -> None:
        """For coordinate roles, derivative/antiderivative stay within CoordRole."""
        try:
            d = role.derivative()
        except TypeCheckError:
            pass  # Terminal role (CoordAcc has no derivative)
        else:
            assert isinstance(d, cxr.AbstractCoordRole)
        try:
            a = role.antiderivative()
        except TypeCheckError:
            pass  # Terminal role (CoordDisp has no antiderivative)
        else:
            assert isinstance(a, cxr.AbstractCoordRole)
