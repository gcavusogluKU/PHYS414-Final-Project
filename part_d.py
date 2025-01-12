import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, CubicSpline

#Returns near evenly spaced data that spans all radii.
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

#Solve_ivp function. Solves coupled ODES of hydrostatic equilibrium.
#Solves for m and ρ. dP/dr is reduced to dρ/dr using pressure equation
def hydrostatic_system(r, y, D, K, q):
    m, ρ = y
    C = 5*K*D**(5/q)/8  #Calculate C from K and given D
    G = 6.6743e-11   #N*m^2*kg^-2
    x = (ρ/D)**(1/q)    #x of P(x)

    dm = 4*np.pi*r**2*ρ
    dρ = -G*np.sqrt(x*x+1)/(8*C*x**5) * q*m*ρ**2/r**2 if r > 0 else 0
    return [dm, dρ]

#Solves the white dwarf system with solve_ivp for given params.
def solve_hydrostatic_system(ρ_c,K,D,ξn,max_step=1e5):
    θ0 = (0,ρ_c) #Since θ(1) = 1 and θ'(0) = 0
    G = 6.6743e-11; q = 3; n = 3/2
    R = np.sqrt(((n+1)*K*ξn**2)/(4*np.pi*G)*ρ_c**(1/n-1))
    solution = solve_ivp(hydrostatic_system, [0, 2*R], θ0, args=(D, K, q), max_step=max_step)
    return solution

#Calculates M,R pairs for given ρc set, and D
def solve_system_D(D,ρc,K,ξn = 3.653753736219,max_step=1e5):
    Ms = 1.988416e30 #Mass of Sun kg
    RE = 6371000     #m, mean radius of earth
    _M = 0*ρc; _R = 0*ρc
    for i in range(len(ρc)):
        solution = solve_hydrostatic_system(ρc[i],K,D,ξn,max_step) #Solve for given ρc, D
        _M[i] = solution.y[0][-1]
        _R[i] = solution.t[-1]
    return _M/Ms, _R/RE #Returns in solar mass units and earth radius units

#Interpolates M', R', and returns total M'(R)-M(R)
def calculate_error(Mexp,Rexp,Mthe,Rthe):
    spline = CubicSpline(Rthe,Mthe) #Interpolate R' and M'
    _Mthe = spline(Rexp)            #Find M'(R) from interpolation
    return np.sum(np.abs(Mexp-_Mthe)) #Return ΣM'(R)-M(R)

#Calculates the total error for given D
#Gets near evenly spaced ρc,M,R from data
#Calculates M', R' from solving IVPs for ρcs
#Interpolates M', R', and returns total M'(R)-M(R)
def find_D_error(D,ρc,M,R,K,max_step=1e5,n_high=14, n_low=12, bias=0):
    #Ms = 1.988416e30 #Mass of Sun kg
    #RE = 6371000     #m, mean radius of earth
    _M, _R, _ρc = get_sample_data(ρc,M,R,n_high,n_low,bias) #Get near evenly spaced data
    _Mthe, _Rthe = solve_system_D(D,_ρc,K,max_step=max_step) #Solve M,R from ρc for D 
    error = calculate_error(_M,_R,_Mthe,_Rthe) #Calculate the error
    return error

#Does bisection for the error calculated by find_D_error.
#Starts with D=1e9 and divides step size to -e when error begins to increase
def bisection_D(ρc,M,R,K,trials=15,max_step=1e5,n_high=14,n_low=12,bias=0):
    fminimize = lambda D: find_D_error(D,ρc,M,R,K,max_step,n_high,n_low,bias)
    D = 1e9
    error = fminimize(D)
    olderror = error
    constant = 1e9
    for i in range(trials):
        olderror = error
        while error <= olderror:
            D += constant
            olderror = error
            error = fminimize(D)
        constant /= -np.e
    return D

def interpolate(_M,_R): #Cubic spline interpolator function
    spline = CubicSpline(_R,_M) 
    return spline  