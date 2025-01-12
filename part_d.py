import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
    return Mtilda, Rtilda

def interpolate(R, M):
    return CubicSpline(R, M)