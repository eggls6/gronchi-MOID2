import numpy as np
import pytest

from gronchi_moid2 import (
    Orbit,
    position_from_eccentric_anomaly,
    state_from_eccentric_anomaly,
    true_anomaly_from_eccentric_anomaly,
)

from conftest import MU_SUN_AU_DAY

ORBITS = [
    Orbit(1.0, 0.0, 0.0, 0.0, 0.0, MU_SUN_AU_DAY),
    Orbit(1.4, 0.23, 0.15, 0.4, 1.1, MU_SUN_AU_DAY),
    Orbit(2.7, 0.8, 2.5, 4.0, 5.5, MU_SUN_AU_DAY),
]
ANOMALIES = np.linspace(0, 2 * np.pi, 13)


@pytest.mark.parametrize("bad", [dict(a=-1.0, e=0.1), dict(a=0.0, e=0.1), dict(a=1.0, e=1.0), dict(a=1.0, e=-0.1)])
def test_orbit_validation(bad):
    with pytest.raises(ValueError):
        Orbit(inc=0, Omega=0, omega=0, mu=1.0, **bad)


@pytest.mark.parametrize("orb", ORBITS)
@pytest.mark.parametrize("u", ANOMALIES)
def test_radius_and_true_anomaly(orb, u):
    r = np.linalg.norm(position_from_eccentric_anomaly(orb, u))
    assert r == pytest.approx(orb.a * (1 - orb.e * np.cos(u)), rel=1e-12)
    f = true_anomaly_from_eccentric_anomaly(orb.e, u)
    assert 0 <= f < 2 * np.pi
    assert r == pytest.approx(orb.a * (1 - orb.e ** 2) / (1 + orb.e * np.cos(f)), rel=1e-10)


@pytest.mark.parametrize("orb", ORBITS)
@pytest.mark.parametrize("u", ANOMALIES)
def test_state_vis_viva_and_angular_momentum(orb, u):
    s = state_from_eccentric_anomaly(orb, u)
    r, v = s[:3], s[3:]
    assert np.allclose(r, position_from_eccentric_anomaly(orb, u))
    rn = np.linalg.norm(r)
    assert v @ v == pytest.approx(orb.mu * (2 / rn - 1 / orb.a), rel=1e-10)
    h = np.cross(r, v)
    assert np.linalg.norm(h) == pytest.approx(np.sqrt(orb.mu * orb.a * (1 - orb.e ** 2)), rel=1e-10)
    # angular momentum points along the orbit normal given by (inc, Omega)
    n = np.array([np.sin(orb.inc) * np.sin(orb.Omega), -np.sin(orb.inc) * np.cos(orb.Omega), np.cos(orb.inc)])
    assert h / np.linalg.norm(h) == pytest.approx(n, abs=1e-12)
