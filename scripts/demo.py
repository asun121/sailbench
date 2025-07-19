import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sailbench.sim.sim_runner_3dof import SimulationRunner


runner = SimulationRunner('configs/default.yaml')
inputs = {'delta_sail': 0.3, 'delta_rudder': 0.0}   # fixed cmds
t, hist = runner.run(inputs)

x, y, psi = hist[:,3], hist[:,4], hist[:,5]
plt.plot(x, y, '-k')
plt.quiver(x[::50], y[::50],
           np.cos(psi[::50]), np.sin(psi[::50]),
           scale=10, width=0.004)
plt.gca().set_aspect('equal')
plt.xlabel('X [m]'); plt.ylabel('Y [m]')
plt.title('3-DOF Proof-of-Concept')
plt.show()
