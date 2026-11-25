"""Reproduce Tables 2 and 3 of Gronchi, Baù & Grassi (2023).

Table 2 prints the elements to only 5-6 significant digits, so the exact critical
points of the printed elements differ from Table 3 by ~1e-3 deg in angle and
up to a few 1e-5 in distance. Tolerances below are set accordingly.
"""

import numpy as np
import pytest

from gronchi_moid2 import moid_gronchi

from conftest import TABLE3

ANG_TOL_DEG = 5e-3
DIST_TOL = 1e-4


def _angdiff(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _kind(k):
    return "saddle" if k.startswith("saddle") else k


@pytest.mark.parametrize("swap", [False, True], ids=["paper-order", "swapped-order"])
def test_table3_all_critical_points(table2_orbits, swap):
    o1, o2 = table2_orbits
    res = moid_gronchi(o2, o1) if swap else moid_gronchi(o1, o2)
    cps = res.critical_points
    assert len(cps) == len(TABLE3) == 10

    found = []
    for c in cps:
        u1, u2 = np.rad2deg(c.u1), np.rad2deg(c.u2)
        found.append((u2, u1, c) if swap else (u1, u2, c))

    for u1_ref, u2_ref, d_ref, kind_ref in TABLE3:
        err, (u1, u2, c) = min(
            ((max(_angdiff(f[0], u1_ref), _angdiff(f[1], u2_ref)), f) for f in found),
            key=lambda x: x[0],
        )
        assert err < ANG_TOL_DEG, f"critical point {u1_ref}, {u2_ref} not found"
        assert c.distance == pytest.approx(d_ref, abs=DIST_TOL)
        assert _kind(c.kind) == kind_ref


def test_table3_moid_is_crossing(table2_orbits):
    res = moid_gronchi(*table2_orbits)
    assert res.moid < 1e-10


def test_table3_morse_and_weierstrass(table2_orbits):
    cps = moid_gronchi(*table2_orbits).critical_points
    kinds = [_kind(c.kind) for c in cps]
    n_min, n_max = kinds.count("minimum"), kinds.count("maximum")
    assert n_min >= 1 and n_max >= 1  # Weierstrass check
    assert len(cps) == 2 * (n_min + n_max)  # Morse check


def test_points_near_u2_equal_pi_are_recovered(table2_orbits):
    """Regression: u2 ~ 180 deg used to be lost through s = tan(u2/2)."""
    o1, o2 = table2_orbits
    for a, b in [(o1, o2), (o2, o1)]:
        u2 = [np.rad2deg(c.u2) for c in moid_gronchi(a, b).critical_points]
        assert any(abs(x - 180.0) < 2.0 for x in u2)
