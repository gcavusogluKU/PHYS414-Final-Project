import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

def fit_WD_data(M,R,minR=1.5):
    def M_fit(R, A, n): #Curve_fit fitting function
        return A*R**((3-n)/(1-n))   #M ∝ R^((3-n)/(1-n))
    
    _M = M[(R > minR)]  #Limit for small R
    _R = R[(R > minR)]  #Limit for small R 

    solution = curve_fit(M_fit, _R, _M, p0=(10,2)) #Fit the data
    A, n = solution[0]  #Get fitting params A,n
    return A, n

def get_int_q(n):
    q = 5*n/(1+n)       #Get q from n
    q = np.rint(q)      #Set q to nearest integer
    return q

def lane_emden_system(ξ, y, n): #Coupled system of ODEs, dθ/dξ = u, du/dξ = -2u/ξ-θ^(3/2)
    θ, u = y
    if ξ == 0: # Avoid division by zero at the center (ξ=0)
        dθ_dξ = 0
        du_dξ = 2/3 - θ**n
    else:
        dθ_dξ = u   # dθ/dξ = u
        du_dξ = -2 / ξ * u - θ**n # du/dξ = -2u/ξ-θ^(3/2)
    return [dθ_dξ, du_dξ]

def solve_lane_emden(method="RK45", ξmax=5, max_step=1e-3, jac=None, atol=1e-6, rtol=1e-3, n=3/2):
    θ0 = (1,0) #Since θ(1) = 1 and θ'(0) = 0
    if jac == None:
        solution = solve_ivp(lane_emden_system, [0, ξmax], θ0, method=method, max_step=max_step, atol=atol, rtol=rtol, args=(n,))
    else:
        solution = solve_ivp(lane_emden_system, [0, ξmax], θ0, method=method, max_step=max_step, atol=atol, rtol=rtol, args=(n,))
    return solution

def calc_central_density(M,R,dθdξn,ξn):
    Ms = 1.988416e30 #Mass of Sun kg
    RE = 6371000     #m, mean radius of earth
    return Ms*M/(4*np.pi*R**3*RE**3)/(-dθdξn/ξn)

def fit_K(ρc,R,ξn,minR=1.5,n=3/2):
    def K_fit(R, A): #Curve_fit fitting function
        return A*R**2
    
    _ρc = ρc[(R > minR)]  #Limit for small R
    RE = 6371000
    _R = R[(R > minR)]*RE #Limit for small R, and multiply with earth radius
    G = 6.6743e-11   #N*m^2*kg^-2

    solution = curve_fit(K_fit, _R, _ρc**(1/n-1), p0=(10,)) #Fit the data
    A = solution[0]  #Get fitting param A
    K = 4*np.pi*G/(ξn**2)/(n+1) * 1/A 
    return K[0]
