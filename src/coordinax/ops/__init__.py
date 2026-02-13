"""Operators to effectuate transformations.

Operators are transformations that can be applied to various objects in
`coordinax`, such as vectors and coordinates. They encapsulate operations like
identity transformations, translations, velocity boosts, and rotations. They
can be combined and simplified. Most powerfully, they can be time dependent.

**Role-Specialized Primitive Operators:**

- ``Translate``: Translates points (acts on ``Point`` role)
- ``Boost``: Boosts velocities (acts on ``PhysVel`` role)
- ``Rotate``: Rotates all spatial components

Let's work through the built-in operations.

>>> import jax
>>> import jax.numpy as jnp
>>> import unxt as u
>>> import coordinax as cx
>>> import coordinax.ops as cxop
>>> import coordinax.frames as cxf

>>> x = jnp.asarray([1.0, 2.0, 3.0])
>>> q = u.Q(x, "km")
>>> vec = cx.Vector.from_(q)
>>> vel = cx.Vector.from_([4, 5, 6], "km/s")
>>> frame = cxf.ICRS()

# Identity Operator

>>> identity_op = cxop.Identity()

An :class:`coordinax.ops.Identity` operator cannot be further simplified:

>>> cxop.simplify(identity_op) is identity_op
True

An :class:`coordinax.ops.Identity` operator leaves the input unchanged:

>>> identity_op(None, x) is x
True
>>> identity_op(None, q) is q
True
>>> identity_op(None, vec) is vec
True
>>> identity_op(None, vel) is vel
True

The inverse of an :class:`coordinax.ops.Identity` operator is itself:

>>> identity_op.inverse is identity_op
True


# Translate Operator

The ``Translate`` operator translates points. It only acts on ``Point`` role vectors.

>>> shift = cxop.Translate.from_([1.0, 1.0, 1.0], "km")
>>> shift
Translate( {'x': Q(1., 'km'), 'y': Q(1., 'km'), 'z': Q(1., 'km')},
          chart=Cart3D() )

Translation with zero delta simplifies to Identity:

>>> zero_shift = cxop.Translate.from_([0.0, 0.0, 0.0], "km")
>>> cxop.simplify(zero_shift)
Identity()

Applying to a Point vector:

>>> shifted = cxop.apply_op(shift, None, vec)
>>> print(shifted)
<Vector: chart=Cart3D, role=Point (x, y, z) [km]
    [2. 3. 4.]>

The inverse negates the translation:

>>> print(shift.inverse)
Translate(
    {'x': Q(-1., 'km'), 'y': Q(-1., 'km'), 'z': Q(-1., 'km')},
    chart=Cart3D()
)


# Boost Operator

The ``Boost`` operator boosts velocities. It only acts on ``PhysVel`` role vectors.

>>> boost = cxop.Boost.from_([10.0, 0.0, 0.0], "km/s")
>>> boost
Boost(
    {'x': Q(10., 'km / s'), 'y': Q(0., 'km / s'), 'z': Q(0., 'km / s')},
    chart=Cart3D()
)

Applying to a PhysVel vector:

>>> boosted = cxop.apply_op(boost, None, vel)


# Rotate Operator

The rotation operator can be initialized from Euler angles:

>>> R = cxop.Rotate.from_euler("z", u.Q(90, "deg"))
>>> print(R.R.round(2))
[[ 0. -1.  0.]
 [ 1.  0.  0.]
 [ 0.  0.  1.]]

Rotations cannot be simplified unless they are identity rotations.

>>> simplified_rot = cxop.simplify(R)
>>> simplified_rot == R
Array(True, dtype=bool)

>>> cxop.simplify(cxop.Rotate(jnp.eye(3)))  # identity rotation
Identity()

An :class:`coordinax.ops.Rotate` operator rotates the input vector:

>>> print(R(x).round(2))
[-2.  1.  3.]

>>> print(R(q).round(2))
Quantity['length']([-2.,  1.,  3.], unit='km')

>>> print(R(vec))
<Vector: chart=Cart3D, role=Point (x, y, z) [km]
    [-2.  1.  3.]>

The inverse of an :class:`coordinax.ops.Rotate` operator is the transpose of the
original rotation matrix:

>>> R_inv = R.inverse
>>> print(R_inv.R.round(2))
[[ 0.  1.  0.]
 [-1.  0.  0.]
 [ 0.  0.  1.]]


# Combining Operators

## Pipe Operator

Operators can be combined using the pipe (`|`) operator, which results in a
:class:`coordinax.ops.Pipe` operator.

>>> combined_op = shift | R
>>> print(jax.tree.map(lambda x: x.round(2), combined_op))
Pipe((
    Translate({'x': Q(1., 'km'), 'y': Q(1., 'km'), 'z': Q(1., 'km')}, chart=Cart3D()),
    Rotate([[ 0. -1.  0.]
            [ 1.  0.  0.]
            [ 0.  0.  1.]])
))

Combined operators can be simplified if possible:

>>> print(cxop.simplify(identity_op | shift))
Translate(
    {'x': Q(1., 'km'), 'y': Q(1., 'km'), 'z': Q(1., 'km')},
    chart=Cart3D()
)


Combined operators have inverses that reverse the order of operations:

>>> combined_inv = combined_op.inverse
>>> combined_inv
Pipe((
    Rotate(f64[3,3](jax)),
    Translate(
        {'x': Q(f64[], 'km'), 'y': Q(f64[], 'km'), 'z': Q(f64[], 'km')}, chart=Cart3D()
    )
))


## Frame Transform Operators

Reference frame transformations are built from primitive operators.
For example, a transformation from one frame to another might include
both a spatial shift and a velocity boost:

>>> shift = cxop.Translate.from_([100_000, 10_000, 0], "km")
>>> boost = cxop.Boost.from_([269_813_212.2, 0, 0], "m/s")
>>> frame_op = shift | boost

This pipeline will translate points and boost velocities. When applied
to a ``PointedVector`` with both position and velocity, both operators act
on their respective fields.


## Galilean Operator

The ``GalileanOp`` is a composite operator for Galilean (non-relativistic)
transformations, combining rotation, translation, and velocity boost.

"""

__all__ = (
    "apply_op",
    "eval_op",
    "simplify",
    # Classes
    "AbstractOperator",
    "AbstractCompositeOperator",
    "Pipe",
    "Identity",
    "Rotate",
    # Role-specialized primitive operators
    "Translate",
    "Boost",
    # Composite
    "GalileanOp",
)

from coordinax import setup_package

with setup_package.install_import_hook("coordinax.ops"):
    from ._src import (
        AbstractCompositeOperator,
        AbstractOperator,
        Boost,
        GalileanOp,
        Identity,
        Pipe,
        Rotate,
        Translate,
        eval_op,
    )
    from coordinax.api import apply_op, simplify

del setup_package
