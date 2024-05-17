# DeepRitz for solving Poisson's eq. in 1D

import scipy
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def energy(u, f):
    N = len(u)+1
    dx = 1/N
    x = np.linspace(0, 1-dx, N)
    u_right = np.hstack((u,np.zeros(1)))
    u_left =  np.hstack((np.zeros(1),u))
    du = (u_right - u_left)/dx
    return 0.5 * np.sum(du**2)*dx - np.sum(f(x)*u_left)*dx

N = 10
x = np.linspace(0,1,N+1)
f = lambda x : np.sin(2*np.pi*x)
u0 = np.zeros(N-1)
res = minimize(lambda u : energy(u, f), u0)

plt.figure(figsize=(4,3))
plt.plot(x, np.hstack(([0],res.x,[0])), 'ko', label="approximate")
x_pts = np.linspace(0,1,100)
plt.plot(x_pts, f(x_pts)/(4*np.pi**2), label="exact")
plt.legend()
plt.show()
