"""
Polarity test for keel
"""

import numpy as np
from sailbench.utils.loader import load_config
from sailbench.foils.keel_3dof import KeelModel3DOF


CFG_PATH = "configs/default.yaml"
CFG      = load_config(CFG_PATH)
KEEL_P   = CFG["keel"]      # param dict for keel


def _keel_force(state):
    """
    Helper that returns [X, Y, N] from the keel
    """
    model = KeelModel3DOF(KEEL_P)
    return model.compute(state)


def test_zero_leeway_has_no_force():
    """
    At zero sway velocity (v) the leeway β≈0,
    so the keel should generate ~zero side-force
    and ~zero yaw moment.
    """
    state = np.array([2.0,     # u  [m/s]
                      0.0,     # v  (no leeway)
                      0.0,     # r
                      0, 0, 0  # x,y,ψ (unused)
                     ])
    X, Y, N = _keel_force(state)
    # Surge drag X_k may be small; focus on Y and N
    tol = 1e-6
    assert abs(Y) < tol, "Keel produced side-force at β≈0"
    assert abs(N) < tol, "Keel produced yaw moment at β≈0"


def test_positive_leeway_force_direction():
    """
    With positive sway (v>0, drifting to starboard),
    β>0. The keel side-force should act to PORT (negative Y)
    to oppose the leeway.
    """
    state = np.array([2.0,      # u
                      0.5,      # v  (starboard drift)
                      0.0,
                      0, 0, 0])
    X, Y, N = _keel_force(state)

    assert Y < 0, "Keel side-force sign does not oppose positive leeway"
    # Because x_k in config is negative (keel aft of CG), a negative Y
    # should yield positive (CCW) yaw moment
    assert N > 0, "Yaw moment sign unexpected for given x_k and Y"
