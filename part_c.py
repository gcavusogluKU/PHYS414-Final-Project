import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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