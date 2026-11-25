# gronchi-moid2

This package computes the Minimum Orbit Intersection Distance (MOID) and **all critical points** of the squared distance between two confocal elliptic orbits. It uses the ordinary-polynomial / eccentric-anomaly method (Sect. 3) of:

> Gronchi, G.F., Baù, G. & Grassi, C. (2023). *Revisiting the computation of the critical points of the Keplerian distance.* Celest. Mech. Dyn. Astron. 135, 48. https://doi.org/10.1007/s10569-023-10161-4

## Installation

```bash
pip install .                 # from the project folder
pip install ".[test]"         # also installs pytest and scipy for the tests
pip install -e ".[test]"      # editable install for development
```

Requires Python ≥ 3.9 and NumPy ≥ 1.22.

## Usage

```python
import numpy as np
from gronchi_moid import Orbit, moid_gronchi

mu = 2.9591221e-4  # au^3/day^2 (only used for the returned velocities)
earth = Orbit(a=1.0, e=0.0167, inc=0.0, Omega=0.0, omega=np.deg2rad(102.9), mu=mu)
ast   = Orbit(a=1.4, e=0.23, inc=0.15, Omega=0.4, omega=1.1, mu=mu)   # angles in rad

res = moid_gronchi(earth, ast)
res.moid            # MOID (same length unit as a)
res.u1, res.u2      # eccentric anomalies of the closest points [rad]
res.f1, res.f2      # true anomalies [rad]
res.state1          # [x, y, z, vx, vy, vz] on orbit 1 at the MOID point
res.critical_points # every critical point, with distance and kind (minimum / maximum / saddle/degenerate)
```

Use any consistent length unit (au, km, …). The algorithm does not depend on scale.

## Tests

```bash
pytest
```

The test suite checks the following:

- **Tables 2 and 3 of the paper:** all 10 critical points are found, with their distances and types, with either orbit given first.
- **Regression for u2 ≈ 180°:** critical points there are recovered.
- **Brute-force comparison:** the MOID matches a grid search plus local minimisation on random NEA-like orbits, both inclined and nearly coplanar.
- **Analytic coplanar case:** for an ellipse and a circle in the same plane, the MOID equals `max(0, q − r, r − Q)`.
- **Nodal distance bound:** the MOID is never larger than the nodal distance.
- **Symmetry and units:** the MOID is the same when the orbits are swapped, and the results are the same in au and km.
- **Geometry:** vis-viva, angular momentum and the conversion between eccentric and true anomaly.

Table 2 gives the orbital elements to only about 5 significant digits. The exact critical points of those printed elements therefore differ from Table 3 by about 1e-3° in angle and up to a few 1e-5 au in distance, and the test tolerances allow for this.

## Changes relative to the original script

- **u2 recovery:** u2 is now found by solving `A cos u2 + B sin u2 + C = 0` for the angle (as in Sect. 3.1 of the paper) rather than through `s = tan(u2/2)`. The old approach lost critical points with u2 ≈ π.
- **Newton step:** each recovered critical point is refined with a Newton step.
- **Residual test:** the threshold is dimensionless (the gradient is divided by `max(a1, a2)**2`).
- **Classification:** minima, maxima and saddles are identified from the analytic Hessian, which does not depend on units.
- **Angular shift:** the shifted computation is also run when the Weierstrass check fails (no minimum or no maximum found), and its results are merged with the unshifted ones.
- **NumPy:** works with NumPy < 2 (`np.arctan` and `np.arctan2` instead of `np.atan` and `np.atan2`).

## Credits

Implementation by Siegfried Eggl (2026), following Gronchi, Baù & Grassi (2023).
