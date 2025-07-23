"""
Polarity test for rudder
"""

import numpy as np
from sailbench.utils.loader import load_config
from sailbench.foils.rudder_3dof import RudderModel3DOF


CFG_PATH = "configs/default.yaml"
CFG      = load_config(CFG_PATH)
RUDDER_P = CFG["rudder"]      # param dict for rudder


def _rudder_force(state, inputs):
    """
    Helper that returns [X, Y, N] from the rudder
    """
    model = RudderModel3DOF(RUDDER_P)
    return model.compute(state, inputs)


def test_rudder_zero_angle():
    state = np.array([2.0, 0.0, 0.0, 0, 0, 0])
    inputs = {"delta_rudder": 0.0}
    _, Y, N = _rudder_force(state, inputs)
    assert abs(Y) < 1
    assert abs(N) < 1