import numpy as np
from scipy.integrate import solve_ivp

def TOV(r, y, K=100): #TOV equations. P reduced as ρ
    m, ν, ρ, mP = y
    dm = 4*np.pi*r**2*ρ
    dν = 2*(m+4*np.pi*K*(ρ**2)*(r**3))/(r*(r-2*m)) if r > 0 else 0
    dρ = -(1+K*ρ)/(4*K)*dν
    dmP = 4*np.pi/np.sqrt(1-2*m/r)*(r**2)*ρ if r > 0 else 0
    return [dm, dν, dρ, dmP]

def stop(r, y, K=100): #Stop condition: do not let ρ be negative
    ρ = y[2]
    return 0 if ρ < 1e-10 else 1
stop.terminal = True

#Solves the Neutron star M and R for given central density
def solve_TOV(ρc,max_step=0.1,K=100):
    y0 = (0,0,ρc,0)
    solution = solve_ivp(TOV, [0, 20], y0, events=stop, max_step=max_step, args=(K,))
    return solution

def obtain_NS_params(ρc,max_step=0.1,K=100): #Function that solves TOV for given central densities
    M = 0*ρc; R = 0*ρc; MP = 0*ρc
    for i in range(len(ρc)):
        solution = solve_TOV(ρc[i],max_step=max_step,K=K)
        M[i] = solution.y[0][-1]
        MP[i] = solution.y[3][-1]
        R[i] = solution.t[-1]
    return M,MP,R

def obtain_max_mass(ρc,K,max_step=0.1): #Function that solves TOV for maximized mass
    Mmax = 0
    for i in range(len(ρc)):
        solution = solve_TOV(ρc[i],max_step=max_step,K=K)
        M_ = solution.y[0][-1]
        if M_ > Mmax:
            Mmax = M_
    return Mmax

## Unused functions for part c.

# def calc_arbitrary_stencils(s, d):      #Function to calculate arbitrary finite difference stencil for dth derivative of stencil s
#     N = len(s)                          #Length of stencil s
#     A = s**np.arange(N)[:,np.newaxis]   #Matrix of powers of s to solve
#     b = np.zeros((N,1))                 #b of linear system Ax=b
#     b[d] = factorial(d)                 #b_d = d!
#     a = np.linalg.solve(A,b).ravel()    #Solve linear system for coefficents a
#     return a

# def numerical_derivative_irregular(fx, x): #Take irregular derivative of irregularly spaced f(x)
#     dfx = 0*fx
#     for i in range(1,len(x)-1):
#         a = calc_arbitrary_stencils([x[i-1]-x[i],0,x[i+1]-x[i]],1) #Calculate stencils for first derivative
#         dfx[i] = np.sum(a*fx[i-1:i+2]) #Apply stencils to array to find irregular derivative
#     a = calc_arbitrary_stencils([0,x[1]-x[0],x[2]-x[0]],1) #Calculate backward stencil for first point
#     dfx[0] = np.sum(a*fx[0:3]) #Apply backward stencil for first point
#     a = calc_arbitrary_stencils([x[-3]-x[-1],x[-2]-x[-1],0],1) #Calculate forward stencil for last point
#     dfx[-1] = np.sum(a*fx[-3:]) #Apply forward stencil for last point
#     return dfx
