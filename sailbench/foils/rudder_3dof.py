import numpy as np
import aerosandbox as asb
from pathlib import Path

from sailbench.foils.base_foil import FoilModel

class RudderModel3DOF(FoilModel):
    def __init__(self, p: dict):
        self.p = p
        airfoil_name = p.get("airfoil_name", "NACA0012")
        self.foil = asb.Airfoil(name=airfoil_name)

        cache = Path(f"cached_foils/{airfoil_name}_rudder_polar.json")
        if not cache.exists():
            self.foil.generate_polars(
                alphas=np.arange(-25, 26, 2),
                Res=np.array([1e6]),
                cache_filename=cache
            )
        else:
            self.foil.generate_polars(cache_filename=cache)

    # ------------------------------------------------------------------
    def compute(self, state: np.ndarray, inputs: dict) -> np.ndarray:
        u, v = state[0], state[1]
        V = np.hypot(u, v) + 1e-9

        beta = np.degrees(np.arctan2(v, u + 1e-9))     # deg
        delta = np.degrees(inputs.get("delta_rudder", 0.0))  # rad→deg

        delta_max = self.p["delta_max_deg"]
        delta = np.clip(delta, -delta_max, delta_max)

        alpha = beta - delta                           # AoA at rudder (deg)

        CL = self.foil.CL_function(alpha=alpha, Re=5e5, mach=0)
        CD = self.foil.CD_function(alpha=alpha, Re=5e5, mach=0)

        q = 0.5 * self.p["rho_water"] * V ** 2
        L = q * self.p["S_r"] * CL
        D = q * self.p["S_r"] * CD

        Y_r = -L                                       # oppose leeway
        X_r = -D
        N_r = Y_r * self.p["x_r"]

        return np.array([X_r, Y_r, N_r])