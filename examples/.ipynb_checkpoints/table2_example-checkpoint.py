"""Reproduce Table 3 of Gronchi, Baù & Grassi (2023) from the elements in Table 2."""

import numpy as np

from gronchi_moid2 import Orbit, moid_gronchi

mu = 2.9591221e-4  # au^3 / day^2

# Table 2 gives cometary elements (q, e, i, Omega, omega); a = q / (1 - e)
orbit1 = Orbit(a=0.16582 / (1 - 0.84577), e=0.84577, inc=0.0, Omega=0.0,
               omega=np.deg2rad(9.09466), mu=mu)
orbit2 = Orbit(a=1.0 / (1 - 0.2), e=0.2, inc=0.0, Omega=0.0,
               omega=np.deg2rad(10.0), mu=mu)

res = moid_gronchi(orbit1, orbit2)
print(f"MOID = {res.moid:.12f} au   (used_shift = {res.used_shift})")
print(f"{'u1 [deg]':>13} {'u2 [deg]':>13} {'d [au]':>11}  type")
for cp in res.critical_points:
    print(f"{np.rad2deg(cp.u1):13.7f} {np.rad2deg(cp.u2):13.7f} {cp.distance:11.7f}  {cp.kind}")
