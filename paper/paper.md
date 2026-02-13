---
title:
  "coordinax: A Python package enabling calculations with coordinates in JAX"
tags:
  - Python
  - Numerical Computing
  - Scientific Computing
authors:
  - name: Nathaniel Starkman
    orcid: 0000-0003-3954-3291
    affiliation: "1,2"
    corresponding: true
  - name: Adrian M. Price-Whelan
    orcid: 0000-0003-0872-7098
    affiliation: "2"
affiliations:
  - index: 1
    name:
      Brinson Prize Fellow at Kavli Institute for Astrophysics and Space
      Research, Massachusetts Institute of Technology, USA
    ror: 042nb2s44
  - index: 2
    name: Department of Physics, Case Western Reserve University, USA
    ror: 051fd9666
  - index: 3
    name: Center for Computational Astrophysics, Flatiron Institute, USA
    ror: 00sekdz59
date: October 1 2025
bibliography: paper.bib
---

# Summary

`coordinax` is a Python package for coordinate and frame-aware computing with
JAX [@jax:18], a high-performance numerical library that supports automatic
differentiation and just-in-time compilation across multiple compute
architectures. `coordinax` is built on top of `quax` [@quax:23] -- a framework
for building JAX-compatible array-like objects -- and `unxt` [@unxt:2025] -- for
handling units and quantities in JAX. With these foundations, coordinax enables
the definition and transformation of coordinate representations, the
construction of coordinate frames and frame transformations, and computations
with coordinates while preserving their associated representations and frames.
In addition, coordinax supports time-dependent operators on coordinates,
allowing for straightforward definition of inertial and non-inertial reference
frames. By providing seamless integration into JAX, coordinax substantially
extends JAX’s capabilities for scientific applications that require rigorous and
efficient handling of coordinates.

Scientific research frequently requires transforming between coordinate systems
and defining reference frames that may be time-dependent or non-inertial.
`coordinax` provides a principled framework within JAX to represent and carry
out such operations consistently and efficiently. By leveraging JAX’s automatic
differentiation, it enables thorough support for these transformations,
including those involving rotating reference frames, while preserving
compatibility with gradient-based methods. At the same time, `coordinax` is
designed to be intuitive and performant, making it straightforward to
incorporate frame-aware computations into existing JAX workflows.

`coordinax` is designed to be accessible to researchers and developers, offering
an intuitive interface for defining, transforming, and analyzing coordinate
frames and their associated data. It supports both straightforward use cases and
more advanced workflows, making it suitable for a broad community of users.
Furthermore, `coordinax` leverages multiple dispatch to enable deep
interoperability with other libraries, currently including `astropy`, and to
accommodate custom array-like objects within JAX. This extensibility ensures
that `coordinax` can serve as a foundation for diverse scientific and
engineering applications, wherever coordinate- and frame-aware computations are
required.

# Statement of Need

JAX is a powerful tool for high-performance numerical computing, featuring
automatic differentiation, just-in-time compilation, and support for sharding
computations. It excels in providing unified interfaces to various compute
architectures, including CPUs, GPUs, and TPUs, to accelerate code execution
[@jax:18]. While JAX supports PyTrees -- nested containers of independent arrays
with some built -- in and user-extendable functionality—there is no intrinsic
notion that the contained arrays are collectively an array-like object. As a
result, operations in JAX remain fundamentally array-based, which poses
challenges for scientific applications that require structured objects such as
coordinate data.

Astropy has been an invaluable resource for the scientific community, with over
10,000 citations to its initial paper and more than 2,000 citations to its 2022
paper [@astropy:13; @astropy:22]. One of its core sub-packages,
`astropy.coordinates`, provides a rich framework for defining coordinate
representations and transforming between reference frames. This functionality
ensures that scientific calculations involving positions, velocities, and other
kinematic quantities are expressed consistently within well-defined frames.
However, despite JAX’s NumPy-like API, it does not provide native support for
such structured objects, and `astropy.coordinates` cannot be directly extended
to work with JAX. This gap highlights the need for a solution that integrates
the expressive power of Astropy’s coordinate framework with the high-performance
and differentiable computing features of JAX.

`coordinax` addresses this gap by providing a function-oriented
framework—consistent with the style of JAX—for handling coordinates,
representations, and frame transformations, with an object-oriented front end
that will be familiar to users of `astropy.coordinates`. By leveraging `quax`
and `unxt`, `coordinax` defines coordinate classes that integrate directly with
JAX functions. Through automatic differentiation, it not only supports
transformations of positions but also propagates velocities and accelerations
through those same transformations. Coordinates can further be aggregated into
phase-space objects, ensuring that positions, velocities, and higher-order
kinematics transform cleanly across inertial and non-inertial frames. This
design allows researchers to work directly with coordinate arrays while relying
on `coordinax` to manage the underlying transformations, preserving both
performance and conceptual clarity.

# Related Works

`astropy.coordinates` provides a framework for coordinate representations,
reference frames, and transformations [@astropy:13;@astropy:22], and `coordinax`
builds on many of the same ideas. Its architecture differs in several key ways:
it integrates with JAX to support time-dependent operators, automatic
differentiation through transformations, and consistent propagation of
velocities and accelerations across inertial and non-inertial frames. The
transformation system is redesigned around multiple dispatch, making it easier
to add new representations, frames, and transforms while remaining compatible
with JAX’s compilation and differentiation. `coordinax` is both a JAX-oriented
framework for coordinate-aware computing and a test-bed for features that will
guide future development in `astropy.coordinates`, even as the projects follow
different implementation paths and timescales.

# Acknowledgements

Support for this work was provided by The Brinson Foundation through a Brinson
Prize Fellowship grant.

The authors thank the Astropy collaboration and many contributors for their work
on `astropy`, which has been invaluable to the scientific community. Members of
the `coordinax` development team are also core developers and maintainers of the
`astropy` package, and we had `astropy` as our guiding star while developing
`coordinax`. We also extend our gratitude to Patrick Kidger for his valuable
communications and guidance on using `quax` to ensure seamless integration of
`coordinax` with `jax`.

# References
