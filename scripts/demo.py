#!/usr/bin/env python3
import matplotlib.pyplot as plt
from sailbench.sim.sim_runner import SimulationRunner

def main():
    runner = SimulationRunner('configs/default.yaml')
    inputs = {'delta_sail': 0.3, 'delta_rudder': 0.0}
    times, hist = runner.run('configs/default.yaml', inputs)
    x, y, psi = hist[:,3], hist[:,4], hist[:,5]

    plt.figure()
    plt.plot(x, y, '-k', label='Path')
    plt.quiver(x[::50], y[::50],
               np.cos(psi[::50]), np.sin(psi[::50]),
               scale=10, width=0.005)
    plt.axis('equal')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('3-DOF Boat Trajectory')
    plt.legend()
    plt.show()

if __name__=='__main__':
    main()
