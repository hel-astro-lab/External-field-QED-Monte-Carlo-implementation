import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import scipy.integrate as integrate
import scipy.special as special
import mpmath
import matplotlib as mpl
from scipy.optimize import minimize
import matplotlib.colors as col


# The auxiliary function of the Breit-Wheeler process
# x = photon quantum parameter
def T_BW(x):

    coeff = 1 / (np.pi * np.sqrt(3) * x**2)

    def xi(a):
        return (2*x) / (3*a*(x-a))
    

    def i1(s):
        def integrand1(p):
            return special.kv(1/3,p)
        return integrate.quad(integrand1, s, np.inf)[0]


    def integrand(s):
        part1 = i1(xi(s))
        part2 = (2 - 1.5 * x * xi(s)) * special.kv(2/3,xi(s))
        return part1 - part2
    
    try:
        integral, error = integrate.quad(integrand, 0, x)
    except Exception:
        return None        
        
    return coeff * integral


# The pair-production rate spectrum
# # x = electron quantum parameter, g = photon quantum parameter
def N_BW_spectrum(g, x):
    coeff = 1 / (np.pi * np.sqrt(3) * g)

    xi = 2 * g / (3*x*(g - x))
    
    def i1(s):
        def integrand1(p):
            return special.kv(1/3,p)
        return integrate.quad(integrand1, s, np.inf)[0]
    
    part1 = i1(xi)
    part2 = (2 - 1.5 * g * xi) * special.kv(2/3, xi)

    return coeff * (part1 - part2)


# The spectrum integrated from 0 to some electron quantum parameter e
# if e = x, where x is the photon quantum parameter, this equals the auxiliary function
def N_BW(x, e): 

    coeff = 1 / (np.pi * np.sqrt(3) * x**2)

    def xi(a):
        return (2*x) / (3*a*(x - a))
    

    def i1(s):
        def integrand1(p):
            return special.kv(1/3,p)
        return integrate.quad(integrand1, s, np.inf)[0]


    def integrand(s):
        part1 = i1(xi(s))
        part2 = (2 - 1.5 * x * xi(s)) * special.kv(2/3,xi(s))
        return part1 - part2
    
    try:
        integral, error = integrate.quad(integrand, 0, e)
    except Exception:
        return None    
        
    return coeff * integral


# The exact cumulative probability of synchrotron radiation
# x = photon quantum parameter, r = ratio
# Here T is the auxiliary function, use the exact value, but calculate it outside this function to avoid 
# calculating it repeatedly when going through different r, as T only depends on chivalue and is time consuming to calculate
def P_BWExct(x, r, T):
    return N_BW(x, x * r)/T #/ T_BW(x)


# The approximation for the auxiliary function
def T_BW_approx(x):
    return 0.37779171*np.exp(-8/(3*x))*x**(-0.333) *(1 - np.exp(-(2.003260712099797)*x**(0.2688)))*(1 - 0.6330922*np.exp(-(3.93231893 + np.log(x))**2 / 29.94554209))


# Functions for the a, b, and n parameters needed in the cumulative probability and pair-production spectrum approximations, and also the solution to the electron quantum parameters
def n_parameter(x):
    A = 14.3129740
    B = 250.285446
    q = 0.571261959
    C = 0.142285910
    D = 4.41463804
    p = 0.63896860
    return 0.25 + A/(1 + B*((x/100)**q + (x/100)**(2*q))) + C*np.exp(-D*(x/1000)**p)

def a_parameter(x):
    A = 0.14549869
    B = 2.10323323
    q = 0.2881441
    p = 0.23634402
    C = 1.89598777
    D = 0.74460445
    return (A*(x/150)**(-q) + B*(x/150)**p -1) * (1 + C*(x/150)**(-D))

def b_parameter(x):
    A = 6.74437759
    B = 2.37682806
    q = 1.56169871
    p = 1.59943553
    C = 0.13741977
    D = 2.75389506
    return -1.033388 - (1 + A*(x/100)**(-p))/(1 + B*(x/100)**(-q)) - C*x**(-D)


# Approximation for the cumulative probability
# chi = photon quantum parameter, r = chi_electron / chi_gamma
def P_BW_approx(chi, r):
    a = a_parameter(chi)
    b = b_parameter(chi)
    n = n_parameter(chi)
    c = 2**(-2*n)*(2*a - 2**(4*n) - 2**(3*n)*b)
    denominator = 1 + b*(1-r)**n + c*(1-r)**(2*n)
    return 1 - (a*(1 - r)**(4*n)) / denominator


# Approximation for the pair-production spectrum
def N_BW_approx(chi, r, T):
    # Here T is the auxiliary function, use the exact value, but calculate it outside this function to avoid 
    # calculating it repeatedly when going through different r, as T only depends on chivalue and is time consuming to calculate
    a = a_parameter(chi)
    b = b_parameter(chi)
    n = n_parameter(chi)
    c = 2**(-2*n)*(2*a - 2**(4*n)-2**(3*n)*b)
    numerator = (4 + 3*b*(1-r)**n + 2*c*(1-r)**(2*n))*(1-r)**(4*n-1)
    denominator = 1 + b*(1-r)**n + c*(1-r)**(2*n)
    return T*a*n*numerator/denominator**2


# Ensures that cubic root returns a real number
def curoot(x):
    if x < 0:
        return -(abs(x))**(1/3)
    else:
        return x**(1/3)


# The functions needed for the solution of the electron quantum parameter ratio, that is r_electron:

def p(z, chi):
    a = a_parameter(chi)
    b = b_parameter(chi)
    n = n_parameter(chi)
    c = (2*a - 2**(4*n) - 2**(3*n)*b) * 2**(-2*n)
    p1 = 12*(1 - z)/a
    p2 = ((1 - z) * c / a)**2
    return (p1 - p2) / 12

def q(z, chi):
    a = a_parameter(chi)
    b = b_parameter(chi)
    n = n_parameter(chi)
    c = (2*a - 2**(4*n) - b*2**(3*n)) * 2**(-2*n)
    q1 = 2*((1 - z) * c / a)**3
    q2 = 2*36 * (1 - z)**2 * c / a**2
    q3 = -27*((1 - z)/a)**2 * (b**2)
    return (q1 + q2 + q3)/216

def determinant(z, chi):
    return (q(z, chi)**2)/4 + (p(z, chi)**3)/27

# Solution to the resolvent cubic equation (Cardano's method)
def root0(z, chi):
    a = a_parameter(chi)
    b = b_parameter(chi)
    n = n_parameter(chi)
    c = (2*a - 2**(4*n) - 2**(3*n)*b) * 2**(-2*n)
    t1 = curoot(-q(z, chi)/2 + (determinant(z, chi))**(1/2))
    t2 = curoot(-q(z, chi)/2 - (determinant(z, chi))**(1/2))
    y = t1 + t2 - (1 - z)*c / (6 * a)
    return y

# Solution u_2^+ of the quartic equation
def root2(z, chi):
    a = a_parameter(chi)
    b = b_parameter(chi)
    n = n_parameter(chi)
    c = (2*a - 2**(4*n) - 2**(3*n)*b) * 2**(-2*n)
    y = root0(z, chi)
    x1 = -np.sqrt(2*y + c*(1-z)/a)
    x2 = np.sqrt(-2*y + c*(1 - z)/a - 2*(b*(1 - z)/a) / np.sqrt(2*y + c*(1 - z)/a))
    return (x1 + x2)/2



# The ratio of quantum parameter: r = chi_electron/chi_gamma taken from the initial photon by a produced electron for z>=0.5
# For z < 0.5, get the ratio by r = 1 - r_electron(1 - z, chi) (see manual symmetrization)
# Multiply by chi_gamma to get chi_electron of the produced electron
def r_electron(z, chi):
    return 1 - root2(z, chi)**(1/n_parameter(chi))



