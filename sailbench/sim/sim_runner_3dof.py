import numpy as np, yaml
from sailbench.solvers.rk4 import rk4_step
from sailbench.dynamics.linear_drag_hydro_3dof import LinearDragModel
from sailbench.foils.dummy_aero_3dof import DummyAeroModel

class SimulationRunner:
    def __init__(self, cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.dt   = cfg['simulation']['dt']
        self.T    = cfg['simulation']['t_final']
        self.prm  = cfg['boat']
        self.hydro = LinearDragModel()
        self.aero  = DummyAeroModel()

    # ------------- ODE -------------
    def _deriv(self, state, inputs):
        u, v, r, x, y, psi = state
        Fh = self.hydro.compute(state, inputs, None, self.prm)
        Fa = self.aero .compute(state, inputs, None, self.prm)
        m, Iz = self.prm['m'], self.prm['Iz']
        cor = np.array([ m*v*r, -m*u*r, 0.0 ])
        X, Y, N = Fh + Fa + cor
        du, dv  = X/m, Y/m
        dr      = N/Iz
        dx      = u*np.cos(psi) - v*np.sin(psi)
        dy      = u*np.sin(psi) + v*np.cos(psi)
        dpsi    = r
        return np.array([du, dv, dr, dx, dy, dpsi])

    # ------------- loop -------------
    def run(self, inputs):
        times   = np.arange(0, self.T+self.dt, self.dt)
        state   = np.zeros(6)
        hist    = np.zeros((len(times), 6))
        for i, _ in enumerate(times):
            hist[i] = state
            state   = rk4_step(self._deriv, state, self.dt, inputs)
        return times, hist
