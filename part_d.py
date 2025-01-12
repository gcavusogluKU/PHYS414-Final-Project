import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, CubicSpline

def get_sample_data(ρc, M, R, n_high=14, n_low=12, bias=0):
    ρcmin = np.log10(np.min(ρc))
    ρcmax = np.log10(np.max(ρc))
    ρci = (ρcmin+ρcmax)/2 + bias
    space = np.concatenate((np.logspace(ρcmin,ρci,n_low)[:-1],np.logspace(ρci,ρcmax,n_high)))

    samples = 0*space
    for i in range(len(space)):
        samples[i] = ρc[np.abs(space[i] - ρc) == np.min(np.abs((space[i] - ρc)))][0]
    samples = np.unique(samples)

    Rtilda = R[np.isin(ρc,samples)]
    Rtilda, sorter = np.unique(Rtilda, return_index=True)
    Mtilda = M[np.isin(ρc,samples)][sorter]
    ρtilda = ρc[np.isin(ρc,samples)][sorter]
    return Mtilda, Rtilda, ρtilda

def hydrostatic_system(r, y, D, K, q):
    m, ρ = y
    C = 5*K*D**(5/q)/8
    G = 6.6743e-11   #N*m^2*kg^-2
    x = (ρ/D)**(1/q)

    dm = 4*np.pi*r**2*ρ
    dρ = -G*np.sqrt(x*x+1)/(8*C*x**5) * q*m*ρ**2/r**2 if r > 0 else 0
    return [dm, dρ]

def solve_hydrostatic_system(ρ_c,K,D,ξn,max_step=1e5):
    θ0 = (0,ρ_c) #Since θ(1) = 1 and θ'(0) = 0
    G = 6.6743e-11; q = 3; n = 3/2
    R = np.sqrt(((n+1)*K*ξn**2)/(4*np.pi*G)*ρ_c**(1/n-1))
    solution = solve_ivp(hydrostatic_system, [0, 2*R], θ0, args=(D, K, q), max_step=max_step)
    return solution

def solve_system_D(D,ρc,K,ξn = 3.653753736219):
    _M = 0*ρc; _R = 0*ρc
    for i in range(len(ρc)):
        solution = solve_hydrostatic_system(ρc[i],K,D,ξn)
        _M[i] = solution.y[0][-1]
        _R[i] = solution.t[-1]
    return _M/1.988416e30, _R/6371000