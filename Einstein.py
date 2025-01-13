import numpy as np
from scipy.integrate import solve_ivp

def TOV(r, y, K=100): #TOV equations. P reduced as ρ
    m, ν, ρ = y
    dm = 4*np.pi*r**2*ρ
    dν = 2*(m+4*np.pi*K*(ρ**2)*(r**3))/(r*(r-2*m)) if r > 0 else 0
    dρ = -(1+K*ρ)/(4*K)*dν
    return [dm, dν, dρ]

def stop(r, y): #Stop condition: do not let ρ be negative
    ρ = y[2]
    return 0 if ρ < 1e-10 else 1
stop.terminal = True

#Solves the Neutron star M and R for given central density
def solve_TOV(ρc,max_step=0.1):
    y0 = (0,0,ρc)
    solution = solve_ivp(TOV, [0, 20], y0, events=stop, max_step=max_step)
    return solution

def obtain_NS_params(ρc,max_step=0.1): #Function that solves TOV for given central densities
    M = 0*ρc
    R = 0*ρc
    for i in range(len(ρc)):
        solution = solve_TOV(ρc[i],max_step=max_step)
        M[i] = solution.y[0][-1]
        R[i] = solution.t[-1]
    return M,R