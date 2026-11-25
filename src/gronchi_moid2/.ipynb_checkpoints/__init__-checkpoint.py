"""Minimum Orbit Intersection Distance (MOID) for two confocal elliptic orbits.

Implements the ordinary-polynomial / eccentric-anomaly method (Sect. 3) of
Gronchi, G.F., Baù, G. and Grassi, C. (2023), "Revisiting the computation of the
critical points of the Keplerian distance", Celest. Mech. Dyn. Astron. 135, 48.
"""

from .core import (
    CriticalPoint,
    MOIDResult,
    Orbit,
    distance_squared,
    moid_gronchi,
    position_from_eccentric_anomaly,
    state_from_eccentric_anomaly,
    true_anomaly_from_eccentric_anomaly,
)

__version__ = "0.1.0"

__all__ = [
    "Orbit",
    "CriticalPoint",
    "MOIDResult",
    "moid_gronchi",
    "position_from_eccentric_anomaly",
    "state_from_eccentric_anomaly",
    "distance_squared",
    "true_anomaly_from_eccentric_anomaly",
    "__version__",
]
