"""Tests for AbstractLinearBasis, CoordinateBasis, PhysicalBasis."""

__all__: tuple[str, ...] = ()

import jax

import coordinax.main as cx
import coordinax.representations as cxr


class TestAbstractLinearBasis:
    """AbstractLinearBasis is a proper subclass of AbstractBasis."""

    def test_coord_basis_inherits(self) -> None:
        """CoordinateBasis is a subclass of AbstractLinearBasis."""
        assert issubclass(cxr.CoordinateBasis, cxr.AbstractLinearBasis)

    def test_phys_basis_inherits(self) -> None:
        """PhysicalBasis is a subclass of AbstractLinearBasis."""
        assert issubclass(cxr.PhysicalBasis, cxr.AbstractLinearBasis)

    def test_abstractlinearbasis_is_abstractbasis(self) -> None:
        """AbstractLinearBasis is a subclass of AbstractBasis."""
        assert issubclass(cxr.AbstractLinearBasis, cxr.AbstractBasis)

    def test_nobasis_not_abstractlinearbasis(self) -> None:
        """NoBasis is NOT a subclass of AbstractLinearBasis."""
        assert not issubclass(cxr.NoBasis, cxr.AbstractLinearBasis)


class TestCoordinateBasis:
    """CoordinateBasis construction, equality, exports."""

    def test_construction(self) -> None:
        """CoordinateBasis() can be constructed."""
        b = cxr.CoordinateBasis()
        assert isinstance(b, cxr.CoordinateBasis)

    def test_singleton_is_coordinatebasis(self) -> None:
        """`coord_basis` is the canonical CoordinateBasis() instance."""
        assert isinstance(cxr.coord_basis, cxr.CoordinateBasis)

    def test_canonical_name(self) -> None:
        """`coord_basis` has the correct canonical name."""
        assert cxr.CoordinateBasis.canonical_name == "coord_basis"

    def test_equality(self) -> None:
        """Two CoordinateBasis() instances are equal."""
        assert cxr.CoordinateBasis() == cxr.CoordinateBasis()

    def test_inequality_with_phys_basis(self) -> None:
        """CoordinateBasis is not equal to PhysicalBasis."""
        assert cxr.coord_basis != cxr.phys_basis

    def test_inequality_with_no_basis(self) -> None:
        """CoordinateBasis is not equal to NoBasis."""
        assert cxr.coord_basis != cxr.no_basis

    def test_jax_static(self) -> None:
        """CoordinateBasis is a valid JAX static value."""

        @jax.jit
        def identity(x):
            return x

        result = identity(cxr.coord_basis)
        assert result == cxr.coord_basis

    def test_exported_from_main(self) -> None:
        """CoordinateBasis and coord_basis exported from coordinax.main."""
        assert hasattr(cx, "CoordinateBasis")
        assert hasattr(cx, "coord_basis")
        assert hasattr(cx, "AbstractLinearBasis")


class TestPhysicalBasis:
    """PhysicalBasis construction, equality, exports."""

    def test_construction(self) -> None:
        """PhysicalBasis() can be constructed."""
        b = cxr.PhysicalBasis()
        assert isinstance(b, cxr.PhysicalBasis)

    def test_singleton_is_physicalbasis(self) -> None:
        """`phys_basis` is the canonical PhysicalBasis() instance."""
        assert isinstance(cxr.phys_basis, cxr.PhysicalBasis)

    def test_canonical_name(self) -> None:
        """`phys_basis` has the correct canonical name."""
        assert cxr.PhysicalBasis.canonical_name == "phys_basis"

    def test_equality(self) -> None:
        """Two PhysicalBasis() instances are equal."""
        assert cxr.PhysicalBasis() == cxr.PhysicalBasis()

    def test_jax_static(self) -> None:
        """PhysicalBasis is a valid JAX static value."""

        @jax.jit
        def identity(x):
            return x

        result = identity(cxr.phys_basis)
        assert result == cxr.phys_basis

    def test_exported_from_main(self) -> None:
        """PhysicalBasis and phys_basis exported from coordinax.main."""
        assert hasattr(cx, "PhysicalBasis")
        assert hasattr(cx, "phys_basis")
