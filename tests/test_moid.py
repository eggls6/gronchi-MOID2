import numpy as np
import pytest

from gronchi_moid2 import Orbit, distance_squared, moid_gronchi

from conftest import MU_SUN_AU_DAY

AU_KM = 1.495978707e8


def brute_force_moid(o1, o2, n=360):
    """Grid search followed by local minimisation from the best cells."""
    minimize = pytest.importorskip("scipy.optimize").minimize
    from gronchi_moid2 import position_from_eccentric_anomaly as pos

    u = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    X1 = np.array([pos(o1, x) for x in u])
    X2 = np.array([pos(o2, x) for x in u])
    D = ((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1)
    best = np.inf
    for idx in np.argsort(D, axis=None)[:40:4]:
        i, j = np.unravel_index(idx, D.shape)
        r = minimize(lambda v: distance_squared(o1, o2, v[0], v[1]), [u[i], u[j]],
                     method="Nelder-Mead",
                     options=dict(xatol=1e-12, fatol=1e-22, maxiter=4000))
        best = min(best, r.fun)
    return float(np.sqrt(max(best, 0.0)))


def _random_orbits(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        q = rng.uniform(0.1, 1.3)
        e = rng.uniform(0.0, 0.95)
        a = q / (1 - e)
        if a > 6:
            continue
        inc = rng.uniform(0, 40) if len(out) % 3 else rng.uniform(0, 1e-3)
        out.append(Orbit(a, e, np.deg2rad(inc), rng.uniform(0, 2 * np.pi),
                         rng.uniform(0, 2 * np.pi), MU_SUN_AU_DAY))
    return out


@pytest.mark.parametrize("orbit", _random_orbits(40, seed=1))
def test_moid_matches_brute_force(earth, orbit):
    res = moid_gronchi(earth, orbit)
    ref = brute_force_moid(earth, orbit)
    # MOID may never exceed a sampled distance, and should match the refined one
    assert res.moid <= ref + 1e-8
    assert res.moid == pytest.approx(ref, abs=1e-6)


@pytest.mark.parametrize("orbit", _random_orbits(15, seed=7))
def test_swap_symmetry(earth, orbit):
    r1 = moid_gronchi(earth, orbit)
    r2 = moid_gronchi(orbit, earth)
    assert r1.moid == pytest.approx(r2.moid, abs=1e-10)
    assert r1.u1 == pytest.approx(r2.u2, abs=1e-6) or r1.moid < 1e-8


def test_unit_independence(table2_orbits):
    o1, o2 = table2_orbits
    res_au = moid_gronchi(o1, o2)
    k1 = Orbit(o1.a * AU_KM, o1.e, o1.inc, o1.Omega, o1.omega, 1.32712440018e11)
    k2 = Orbit(o2.a * AU_KM, o2.e, o2.inc, o2.Omega, o2.omega, 1.32712440018e11)
    res_km = moid_gronchi(k1, k2)
    assert len(res_km.critical_points) == len(res_au.critical_points)
    for a, b in zip(res_au.critical_points, res_km.critical_points):
        assert b.distance / AU_KM == pytest.approx(a.distance, abs=1e-9)
        assert a.kind == b.kind


@pytest.mark.parametrize("q,e,r2", [(0.5, 0.3, 1.0), (1.5, 0.1, 1.0), (0.2, 0.9, 3.0), (2.0, 0.0, 1.0)])
@pytest.mark.parametrize("omega_deg", [0.0, 37.0, 200.0])
def test_coplanar_circle_analytic(q, e, r2, omega_deg):
    """Coplanar ellipse vs circle of radius r2: MOID = max(0, q - r2, r2 - Q)."""
    ell = Orbit(q / (1 - e), e, 0.0, 0.0, np.deg2rad(omega_deg), MU_SUN_AU_DAY)
    circ = Orbit(r2, 0.0, 0.0, 0.0, 0.0, MU_SUN_AU_DAY)
    if e == 0.0:
        pytest.skip("concentric circles have a continuum of critical points")
    Q = q * (1 + e) / (1 - e)
    expected = max(0.0, q - r2, r2 - Q)
    assert moid_gronchi(ell, circ).moid == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("orbit", _random_orbits(15, seed=3))
def test_moid_not_above_nodal_distance(earth, orbit):
    """d_min <= nodal distance (Gronchi & Niederman 2020)."""
    from gronchi_moid2.core import _frame_vectors

    def r_at_direction(o, nvec):
        P, Q = _frame_vectors(o)
        f = np.arctan2(nvec @ Q, nvec @ P)
        return o.a * (1 - o.e ** 2) / (1 + o.e * np.cos(f))

    h1 = np.cross(*_frame_vectors(earth))
    h2 = np.cross(*_frame_vectors(orbit))
    n = np.cross(h1, h2)
    if np.linalg.norm(n) < 1e-6:
        pytest.skip("nearly coplanar: mutual node undefined")
    n /= np.linalg.norm(n)
    dnod = min(abs(r_at_direction(earth, s * n) - r_at_direction(orbit, s * n)) for s in (1, -1))
    assert moid_gronchi(earth, orbit).moid <= dnod + 1e-9


def test_critical_points_have_small_gradient(earth):
    o = Orbit(1.4, 0.23, 0.15, 0.4, 1.1, MU_SUN_AU_DAY)
    res = moid_gronchi(earth, o)
    assert all(c.grad_norm < 1e-8 for c in res.critical_points)
    kinds = {c.kind for c in res.critical_points}
    assert {"minimum", "maximum"} <= kinds


def test_result_consistency(earth):
    o = Orbit(1.4, 0.23, 0.15, 0.4, 1.1, MU_SUN_AU_DAY)
    res = moid_gronchi(earth, o)
    assert res.moid == pytest.approx(np.linalg.norm(res.state1[:3] - res.state2[:3]), abs=1e-12)
    assert res.moid == min(c.distance for c in res.critical_points)
    assert 0 <= res.f1 < 2 * np.pi and 0 <= res.f2 < 2 * np.pi
