import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

def fit_WD_data(M,R,minR=0.15):
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

def lane_emden_system(ξ, y, n=1.5): #Coupled system of ODEs, dθ/dξ = u, du/dξ = -2u/ξ-θ^(3/2)
    θ, u = y
    if ξ == 0: # Avoid division by zero at the center (ξ=0)
        dθ_dξ = 0
        du_dξ = 2/3 - θ**n
    else:
        dθ_dξ = u   # dθ/dξ = u
        du_dξ = -2 / ξ * u - θ**n # du/dξ = -2u/ξ-θ^(3/2)
    return [dθ_dξ, du_dξ]

def solve_lane_emden(method="RK45", ξmax=5, max_step=1e-3, jac=None, atol=1e-6, rtol=1e-3):
    θ0 = (1,0) #Since θ(1) = 1 and θ'(0) = 0
    if jac == None:
        solution = solve_ivp(lane_emden_system, [0, ξmax], θ0, method=method, max_step=1e-4, atol=atol, rtol=rtol)
    else:
        solution = solve_ivp(lane_emden_system, [0, ξmax], θ0, method=method, max_step=1e-4, atol=atol, rtol=rtol, jac=jac)
    return solution