# Model of a spring mass damper
import numpy as np
import control as ct
import matplotlib.pyplot as plt

# x^dot^dot = 1/m * (f(t) - k * x - d * x^dot)

# (s^2 + d/m*s + k/m)x = 1/m * f(t)
# x = f(t)/m/(s^2 + d/m*s + k/m)
# x = f(t)/m/(s^2 + 2*zeta*o*s + o^2)
# omega = sqrt(k/m)
# 2 * zeta * omega = d/m
# zeta = d/m * sqrt(m/k) / 2
# zeta = d/2/sqrt(mk)

#        x       0     1     x          0
#d/dt (     ) = (          )(     ) + (    ) f(t)
#      x^dot     -k/m  -d/m  x^dot     1/m

def get_kd(m, omega, zeta):
    # omega = sqrt(k/m)
    # k = m * omega^2
    k = m * omega**2

    # zeta = d / 2 / sqrt(mk)
    d = zeta * 2 * np.sqrt(m*k)
    return (k,d)


def get_ABCD_cont(k,m,d):
    A = [[0, 1], [-k/m, -d/m]]
    B = [[0], [1/m]]
    C = [1, 0]
    D = 0
    return (A,B,C,D)

def get_ABCD_disc(k,m,d,Ts):
    (A, B, C, D) = get_ABCD_cont(k,m,d)
    sys = ct.ss(A, B, C, D)
    sys_d = ct.sample_system(sys, Ts)
    return (sys_d.A, sys_d.B, sys_d.C, sys_d.D)

if __name__ == "__main__":
    m = 1
    omega = 1
    zeta = 0.1
    k,d = get_kd(m, omega, zeta)

    (A,B,C,D) = get_ABCD_cont(k,m,d)
    sys = ct.ss(A, B, C, D)
    ct.bode_plot(sys)

    Ts = 0.1
    (A,B,C,D) = get_ABCD_disc(k,m,d,Ts)
    sys_d = ct.ss(A, B, C, D, Ts)
    ct.bode_plot(sys_d)

    plt.show()

