import numpy as np
from .base import AeroModel

class DummyAeroModel(AeroModel):
    def compute(self, state, inputs, env, params):
        # simple constant forward sail force
        return np.array([100.0, 0.0, 0.0])
