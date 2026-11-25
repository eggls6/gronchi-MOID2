import numpy as np
import pytest

from gronchi_moid2 import Orbit

MU_SUN_AU_DAY = 2.9591221e-4

# Table 2 of Gronchi, Baù & Grassi (2023): cometary elements q, e, i, Omega, omega (deg)
TABLE2_Q1, TABLE2_E1, TABLE2_W1 = 0.16582, 0.84577, 9.09466
TABLE2_Q2, TABLE2_E2, TABLE2_W2 = 1.0, 0.2, 10.0

# Table 3: (u1 [deg], u2 [deg], d, type) for orbit 1 / orbit 2 of Table 2
TABLE3 = [
    (116.0625325, 153.9899286, 0.0000000, "minimum"),
    (243.6382848, 203.6865581, 0.0000000, "minimum"),
    (179.8948964, 178.9198966, 0.4845432, "saddle"),
    (1.6247542, 2.0946456, 0.8341185, "minimum"),
    (24.0090191, 38.3799855, 0.8401907, "saddle"),
    (334.2162041, 317.5202237, 0.8445898, "saddle"),
    (324.5270438, 126.4762243, 1.6264123, "saddle"),
    (34.8254033, 231.0377067, 1.6334795, "saddle"),
    (0.9077692, 180.7796090, 1.6658557, "maximum"),
    (179.9346562, 358.9929507, 2.9845260, "maximum"),
]


def cometary(q, e, inc_deg, Om_deg, om_deg, mu=MU_SUN_AU_DAY):
    return Orbit(
        a=q / (1.0 - e), e=e, inc=np.deg2rad(inc_deg),
        Omega=np.deg2rad(Om_deg), omega=np.deg2rad(om_deg), mu=mu,
    )


@pytest.fixture
def table2_orbits():
    o1 = cometary(TABLE2_Q1, TABLE2_E1, 0.0, 0.0, TABLE2_W1)
    o2 = cometary(TABLE2_Q2, TABLE2_E2, 0.0, 0.0, TABLE2_W2)
    return o1, o2


@pytest.fixture
def earth():
    return Orbit(
        a=1.00000011, e=0.01671022, inc=np.deg2rad(5e-5),
        Omega=np.deg2rad(-11.26064), omega=np.deg2rad(102.94719), mu=MU_SUN_AU_DAY,
    )
