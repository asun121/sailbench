import numpy as np
from sailbench.utils.loader import load_config
from sailbench.dynamics.linear_drag_hydro_3dof import LinearDragModel
from sailbench.aero.dummy_aero_3dof  import DummyAeroModel
from sailbench.solvers.rk4 import rk4_step

class SimulationRunner:
    def __init__(self, cfg_path):
        cfg = load_config(cfg_path)
        self.sim    = cfg['simulation']
        self.params = cfg['boat']
        self.env    = {
            'wind_speed': cfg['environment']['wind_speed'],
            'wind_dir':   np.deg2rad(cfg['environment']['wind_dir_deg'])
        }
        self.hydro  = LinearDragModel()
        self.aero   = DummyAeroModel()

    def step(self, state, inputs):
        # 1) hydro & aero
        Fh = self.hydro.compute(state, inputs, self.env, self.params)
        Fa = self.aero.compute (state, inputs, self.env, self.params)
        # 2) Coriolis‐like
        u, v, r = state[0], state[1], state[2]
        Cxu =  self.params['m'] * v * r
        Cyv = -self.params['m'] * u * r
        # 3) total forces [X, Y, N]
        total = Fh + Fa + np.array([Cxu, Cyv, 0.0])
        # 4) build derivative function
        def dyn(s):
            du = total[0] / self.params['m']
            dv = total[1] / self.params['m']
            dr = total[2] / self.params['Iz']
            x, y, psi = s[3], s[4], s[5]
            dx  =  s[0]*np.cos(psi) - s[1]*np.sin(psi)
            dy  =  s[0]*np.sin(psi) + s[1]*np.cos(psi)
            dpsi=  s[2]
            return np.array([du, dv, dr, dx, dy, dpsi])
        return dyn

    def run(self, cfg_path, inputs):
        dt, T = self.sim['dt'], self.sim['t_final']
        times  = np.arange(0, T+dt, dt)
        state  = np.zeros(6)
        history= np.zeros((len(times), 6))
        dyn_f  = None

        for i, t in enumerate(times):
            history[i] = state
            dyn_f = self.step(state, inputs)
            state = rk4_step(dyn_f, t, state, dt)
        return times, history
