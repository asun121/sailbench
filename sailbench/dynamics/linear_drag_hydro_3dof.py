import numpy as np
from sailbench.dynamics.base_hydro import HydroModel

class LinearDragModel(HydroModel):
    """Simple −a·u, −a·v, −a·r linear drag."""

    def compute(self, state, inputs, env, params):
        u, v, r = state[0], state[1], state[2]
        return np.array([
            -params['au'] * u,
            -params['av'] * v,
            -params['ar'] * r
        ])
