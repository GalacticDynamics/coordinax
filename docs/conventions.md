# 📜 Conventions

## Naming Conventions

`coordinax` uses a few conventions to make the code more readable and to avoid
verbosity. Many of these are also found in the [Glossary](glossary.md).

- `Abstract...`: a class that is not meant to be instantiated directly, but
  rather to be subclassed. Abstract classes are prefixed with 'Abstract'.
  Concrete (or 'final') classes are not so prefixed. As a further rule, no
  abstract class inherits from a concrete class and no concrete class inherits
  from any other concrete class.
- `Pos`: a shorthand for "position", used in class names for concision.
- `Vel`: a shorthand for "velocity", used in class names for concision.
- `Acc`: a shorthand for "acceleration", used in class names for concision.

## Functional vs Object-Oriented APIs

As `JAX` is function-oriented, but Python is generally object-oriented,
`coordinax` provides both functional and object-oriented APIs. The functional
APIs are the primary APIs, but the object-oriented APIs are easy to use and call
the functional APIs, so lose none of the power.

As an example, consider the following code snippets:

```{code-block} python

>>> import coordinax.vecs as cxv

>>> q = cxv.CartesianPos3D.from_([1, 2, 3], "m")
>>> print(q)
<CartesianPos3D: (x, y, z) [m]
    [1 2 3]>

```

First we'll show the object-oriented API:

```{code-block} python

>>> q.vconvert(cxv.SphericalPos)
SphericalPos(
  r=Distance(3.7416575, unit='m'),
  theta=Angle(0.64052236, unit='rad'),
  phi=Angle(1.1071488, unit='rad')
)

```

And now the function-oriented API:

```{code-block} python

>>> cxv.vconvert(cxv.SphericalPos, q)
SphericalPos(
  r=Distance(3.7416575, unit='m'),
  theta=Angle(0.64052236, unit='rad'),
  phi=Angle(1.1071488, unit='rad')
)

```

## Multiple Dispatch

`coordinax` uses [multiple dispatch](https://beartype.github.io/plum/) to hook
into `quax`'s flexible and extensible system to enable custom array-ish objects,
like {class}`~unxt.quantity.Quantity`, in `JAX`. Also, `coordinax` uses multiple
dispatch to enable deep interoperability between `coordinax` and other
libraries, like `astropy` (and anything user-defined).

For example, `coordinax` provides a {meth}`~coordinax.vecs.AbstractVector.from_`
method that can convert an `astropy.Representation` to a
{class}`~coordinax.vecs.AbstractVector`:

```{code-block} python

>>> import astropy.coordinates as apyc
>>> import coordinax.vecs as cxv

>>> aq = apyc.CartesianRepresentation([1, 2, 3], unit="m")
>>> aq
<CartesianRepresentation (x, y, z) in m
    (1., 2., 3.)>

>>> xq = cxv.CartesianPos3D.from_(aq)  # unxt Quantity
>>> print(xq)
<CartesianPos3D: (x, y, z) [m]
    [1. 2. 3.]>

```

This easy interoperability is enabled by multiple dispatch, which allows the
{meth}`~coordinax.vecs.AbstractVector.from_` method to dispatch to the correct
implementation based on the types of the arguments.

For more information on multiple dispatch, see the
[plum documentation](https://beartype.github.io/plum/).
