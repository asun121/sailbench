import numpy as np, yaml
from sailbench.solvers.rk4 import rk4_step
from sailbench.dynamics.linear_drag_hydro_3dof import hydro_drag
from sailbench.aero.dummy_aero_3dof import sail_force

class SimulationRunner:
    def __init__(self, cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.dt   = cfg['simulation']['dt']
        self.T    = cfg['simulation']['t_final']
        self.prm  = cfg['boat']

    # -------- core ODE --------
    def f(self, state, inputs):
        u, v, r, x, y, psi = state
        # Forces & moments
        Fh = hydro_drag(state, self.prm)
        Fa = sail_force (state, inputs)
        m, Iz = self.prm['m'], self.prm['Iz']
        # Coriolis terms
        C = np.array([ m*v*r, -m*u*r, 0.0 ])
        X, Y, N = Fh + Fa + C
        # Translational accel
        du, dv = X/m, Y/m
        dr     = N/Iz
        # Kinematics
        dx = u*np.cos(psi) - v*np.sin(psi)
        dy = u*np.sin(psi) + v*np.cos(psi)
        dpsi = r
        return np.array([du, dv, dr, dx, dy, dpsi])

    # -------- main loop --------
    def run(self, inputs):
        times   = np.arange(0, self.T+self.dt, self.dt)
        state   = np.zeros(6)       # [u v r x y psi]
        history = np.zeros((len(times), 6))
        for i, _ in enumerate(times):
            history[i] = state
            state = rk4_step(self.f, state, self.dt, inputs)
        return times, history
