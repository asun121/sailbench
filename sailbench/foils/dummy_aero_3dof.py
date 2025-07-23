import numpy as np
from sailbench.foils.base_foil import FoilModel

class DummyAeroModel(FoilModel):
    """Constant forward thrust proportional to sail trim."""

    def compute(self, state, inputs, env, params):
        thrust = 100.0 * inputs.get('delta_sail', 0.3)
        return np.array([thrust, 0.0, 0.0])
