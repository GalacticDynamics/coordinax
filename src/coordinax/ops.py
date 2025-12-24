"""Operators to effectuate transformations.

Operators are transformations that can be applied to various objects in
`coordinax`, such as vectors and coordinates. They encapsulate operations like
identity transformations, translations, velocity boosts, and rotations. They
can be combined and simplified. Most powerfully, they can be time dependent.

**Role-Specialized Primitive Operators:**

- ``Translate``: Translates points (acts on ``Point`` role)
- ``Boost``: Boosts velocities (acts on ``Vel`` role)
- ``AccelShift``: Shifts accelerations (acts on ``Acc`` role)
- ``Rotate``: Rotates all spatial components

Let's work through the built-in operations.

>>> import jax
>>> import jax.numpy as jnp
>>> import unxt as u
>>> import coordinax as cx
>>> import coordinax.ops as cxo
>>> import coordinax.frames as cxf

>>> x = jnp.asarray([1.0, 2.0, 3.0])
>>> q = u.Q(x, "km")
>>> vec = cx.Vector.from_(q)
>>> vel = cx.Vector.from_([4, 5, 6], "km/s")
>>> frame = cxf.ICRS()

# Identity Operator

>>> identity_op = cxo.Identity()

An :class:`coordinax.ops.Identity` operator cannot be further simplified:

>>> cxo.simplify(identity_op) is identity_op
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

>>> shift = cxo.Translate.from_([1.0, 1.0, 1.0], "km")
>>> shift
Translate(Q(f64[3], 'km'))

Translation with zero delta simplifies to Identity:

>>> zero_shift = cxo.Translate.from_([0.0, 0.0, 0.0], "km")
>>> cxo.simplify(zero_shift)
Identity()

Applying to a Point vector:

>>> shifted = cxo.apply_op(shift, None, vec)
>>> print(shifted)
<Cart3D: (x, y, z) [km]
    [2. 3. 4.]>

The inverse negates the translation:

>>> print(shift.inverse)
Translate(Q(f64[3], 'km'))


# Boost Operator

The ``Boost`` operator boosts velocities. It only acts on ``Vel`` role vectors.

>>> boost = cxo.Boost.from_([10.0, 0.0, 0.0], "km/s")
>>> boost
Boost(Q(f64[3], 'km / s'))

Applying to a Vel vector:

>>> boosted = cxo.apply_op(boost, None, vel)


# AccelShift Operator

The ``AccelShift`` operator shifts accelerations. It only acts on ``Acc`` role
vectors.

>>> accel = cxo.AccelShift.from_([0.0, 0.0, -9.8], "m/s^2")
>>> accel
AccelShift(Q(f64[3], 'm / s2'))


# Rotate Operator

The rotation operator can be initialized from Euler angles:

>>> R = cxo.Rotate.from_euler("z", u.Q(90, "deg"))
>>> print(R.R.round(2))
[[ 0. -1.  0.]
 [ 1.  0.  0.]
 [ 0.  0.  1.]]

Rotations cannot be simplified unless they are identity rotations.

>>> simplified_rot = cxo.simplify(R)
>>> simplified_rot == R
Array(True, dtype=bool)

>>> cxo.simplify(cxo.Rotate(jnp.eye(3)))  # identity rotation
Identity()

An :class:`coordinax.ops.Rotate` operator rotates the input vector:

>>> print(R(x).round(2))
[-2.  1.  3.]

>>> print(R(q).round(2))
Quantity['length']([-2.,  1.,  3.], unit='km')

>>> print(R(vec))
<Vector: (x, y, z) [km]
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
>>> print(combined_op)
Pipe((
    Translate(Q(f64[3], 'km')),
    Rotate([[ 0.         -0.99999994  0.        ]
            [ 0.99999994  0.          0.        ]
            [ 0.          0.          0.99999994]])
))

Combined operators can be simplified if possible:

>>> print(cxo.simplify(identity_op | shift))
Translate(Q(f64[3], 'km'))


Combined operators have inverses that reverse the order of operations:

>>> combined_inv = combined_op.inverse
>>> combined_inv
Pipe((Rotate(R=f32[3,3]), Translate(delta=Q(f64[3], 'km'))))


## Frame Transform Operators

Reference frame transformations are built from primitive operators.
For example, a transformation from one frame to another might include
both a spatial shift and a velocity boost:

>>> shift = cxo.Translate.from_([100_000, 10_000, 0], "km")
>>> boost = cxo.Boost.from_([269_813_212.2, 0, 0], "m/s")
>>> frame_op = shift | boost

This pipeline will translate points and boost velocities. When applied
to a ``FiberPoint`` with both position and velocity, both operators act
on their respective fields.


## Galilean Operator

The ``GalileanOp`` is a composite operator for Galilean (non-relativistic)
transformations, combining rotation, translation, and velocity boost.

"""

__all__ = (
    "apply_op",
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
    "AccelShift",
    # Composite
    "GalileanOp",
)

from .setup_package import RUNTIME_TYPECHECKER, install_import_hook

with install_import_hook("coordinax.ops"):
    from ._src.api import apply_op, simplify
    from ._src.operators import (
        AbstractCompositeOperator,
        AbstractOperator,
        AccelShift,
        Boost,
        GalileanOp,
        Identity,
        Pipe,
        Rotate,
        Translate,
    )

del install_import_hook, RUNTIME_TYPECHECKER
