"""Tests for AbstractSemanticKind and all concrete subclasses."""

__all__: tuple[str, ...] = ()

from hypothesis import given

import coordinax.hypothesis.representations as cxrst
import coordinax.representations as cxr

# ===================================================================


class TestSemanticKindEqualityAndHashing:
    """Instances of the same AbstractSemanticKind subclass are equal and hash-equal."""

    @given(cls=cxrst.semantic_classes())
    def test_same_class_instances_are_equal(self, cls: type) -> None:
        """Two fresh instances of the same class compare equal."""
        assert cls() == cls()

    @given(cls=cxrst.semantic_classes())
    def test_same_class_instances_hash_equal(self, cls: type) -> None:
        """Two fresh instances of the same class have the same hash."""
        assert hash(cls()) == hash(cls())

    @given(sem=cxrst.semantics())
    def test_instances_are_hashable(self, sem: cxr.AbstractSemanticKind) -> None:
        """Every kind instance can be stored in a set and used as a dict key."""
        s = {sem}
        assert sem in s
        d = {sem: 42}
        assert d[sem] == 42

    @given(a=cxrst.semantics(), b=cxrst.semantics())
    def test_equality_iff_same_type(
        self, a: cxr.AbstractSemanticKind, b: cxr.AbstractSemanticKind
    ) -> None:
        """Instances are equal iff they have the same concrete type."""
        assert (a == b) == (type(a) is type(b))

    def test_instance_equal_to_fresh_instance(self) -> None:
        """The canonical instance equals a freshly constructed instance."""
        assert cxr.loc == cxr.Location()

    def test_instance_hash_equal_to_fresh_instance(self) -> None:
        """The canonical instance hashes equal to a freshly constructed instance."""
        assert hash(cxr.loc) == hash(cxr.Location())

    @given(sem=cxrst.semantics())
    def test_usable_as_set_members(self, sem: cxr.AbstractSemanticKind) -> None:
        """Instances can be stored in a set; a duplicate does not grow it."""
        s = {sem}
        assert len(s) == 1
        # Adding a fresh instance of the same type is a duplicate
        s.add(type(sem)())
        assert len(s) == 1
