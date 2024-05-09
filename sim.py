import numpy as np
from spring_mass_damper import get_kd, get_ABCD_disc
import matplotlib.pyplot as plt

# Define the system model
Ts = 0.1
omega = 1
zeta = 0.1
m = 1

k,d = get_kd(m, omega, zeta)
A,B,C,D = get_ABCD_disc(k, m, d, Ts)
n_states = B.shape[0]
n_inputs = B.shape[1]

# Simulation parameters
N = 100
x_0 = np.array([[1], [0]])
assert x_0.shape == (n_states,1)

# Define the input
u = np.zeros((N,n_inputs,1))

# Main loop
x_store = np.zeros((N,n_states, 1))
x_store[0] = x_0
for i in range(1,N):
    x_im1 = x_store[i-1]
    x_i = A @ x_im1 + B * u[i-1]
    x_store[i] = x_i

x_pos = x_store[:,0,0]

ax = plt.figure().subplots(1,1)
time = np.arange(0,N*Ts,Ts)
ax.plot(time, x_pos)
ax.set_xlabel('time [s]')
ax.set_ylabel('position')
plt.show()
