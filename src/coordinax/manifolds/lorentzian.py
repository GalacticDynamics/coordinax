r"""Measurements that need a timelike direction.

The verbs here are gated on the metric having a **Lorentzian** signature
$(-,+,\ldots,+)$ -- exactly one timelike direction -- expressed as the type
`~coordinax.manifolds.AbstractLorentzianMetricField`.  Without such a direction
there is no causal structure: every geodesic_distance has the same character and there
is nothing to classify, no proper time to elapse.

Why this namespace is named for the *signature* and not for "spacetime": the
library already ships a spacetime that is **not** Lorentzian.
`~coordinax.charts.galileanct` is a 4-D Galilean spacetime chart on a
`~coordinax.manifolds.CartesianProductManifold` with a `ProductMetric`, and none
of these verbs apply to it.  A ``spacetime`` namespace would therefore promise
membership its dispatch refuses -- the same dishonesty that used to live in the
dispatch itself, where these functions accepted any chart and rejected it at
runtime.  ``lorentzian`` says exactly what the gate is.

Nor is it named ``minkowski``: the gate is the signature, not the metric.  A
curved spacetime metric -- Schwarzschild, FLRW -- inherits the marker and
acquires every verb here without any change.

This module is a **view**, not a home.  The dispatches live with the manifold
that implements them (Minkowski's in ``_src/minkowski/causality.py``, and a
future Schwarzschild's alongside its own metric), exactly as
`coordinax.transforms.groups` is a view over markers defined elsewhere.

`~coordinax.manifolds.interval` is re-exported here even though it is *not*
gated -- the signed quadratic form is defined for every metric, and
`coordinax.manifolds` remains its canonical home.  It appears here because
`causal_character` is literally the sign of it and `proper_time` the root of it,
so a relativistic workflow wants them all to hand.

Examples
--------
>>> import unxt as u
>>> import coordinax.charts as cxc
>>> import coordinax.manifolds as cxm

>>> o = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
>>> ev = {"ct": u.Q(5.0, "m"), "x": u.Q(1.0, "m"),
...       "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}

>>> cxm.lorentzian.interval(cxc.minkowskict, o, ev).round(2)
Q(-24., 'm2')

>>> cxm.lorentzian.causal_character(cxc.minkowskict, o, ev)
'timelike'

>>> cxm.lorentzian.proper_time(cxc.minkowskict, o, ev).uconvert("ns").round(2)
Q(16.34, 'ns')

A metric with no timelike direction is refused, by name:

>>> a = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}
>>> b = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
>>> try:
...     cxm.lorentzian.causal_character(cxc.cart3d, a, b)
... except NotImplementedError as e:
...     print(str(e).split("--")[0].strip())
causal_character() requires a Lorentzian metric

"""

__all__ = (
    "causal_character",
    "proper_time",
    "proper_distance",
    "rapidity_between",
    # Re-exported, canonical in `coordinax.manifolds`: defined for every metric,
    # but the quantity the three above read.
    "interval",
)

from coordinaxs.api.manifolds import (
    causal_character,
    interval,
    proper_distance,
    proper_time,
    rapidity_between,
)
