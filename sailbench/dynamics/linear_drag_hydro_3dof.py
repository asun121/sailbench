import numpy as np
from .base import HydroModel

class LinearDragModel(HydroModel):
    def compute(self, state, inputs, env, params):
        u, v, r = state[0], state[1], state[2]
        Xh = -params['au'] * u
        Yh = -params['av'] * v
        Nh = -params['ar'] * r
        return np.array([Xh, Yh, Nh])
