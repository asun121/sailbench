import numpy as np

def rk4_step(f, t, state, dt, *args):
    k1 = f(state, *args)
    k2 = f(state + dt*k1/2, *args)
    k3 = f(state + dt*k2/2, *args)
    k4 = f(state + dt*k3,   *args)
    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6