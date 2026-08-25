import numpy as np
import scipy.integrate as integrate
import scipy.special as special


# The Auxiliary function of synchrotron radiation
# x = electron quantum parameter
def T_Synchrotron(x): 

    coeff = -1 / (np.pi * np.sqrt(3) * x**2)

    def xi(a):
        return (2*a) / (3*x*(x-a))
    
    def i1(s):
        def integrand1(p):
                return special.kv(1/3,p)
        return integrate.quad(integrand1, s, np.inf, epsabs=1e-45, epsrel=1e-10)[0]

    def integrand(s):
        part1 = i1(xi(s))
        part2 = (2 + 1.5 * s * xi(s)) * special.kv(2/3,xi(s))
        return part1 - part2
    
    try:
        integral, error = integrate.quad(integrand, 0, x, epsabs=1e-45, epsrel=1e-10)
    except Exception:
        return None        
        
    return coeff * integral


# The photon emission rate spectrum
# x = electron quantum parameter, g = photon quantum parameter
def N_Synch_spectrum(x, g):
    coeff = -1 / (np.pi * np.sqrt(3) * x)

    xi = 2 * g / (3*x*(x - g))
    
    def i1(s):
        def integrand1(p):
                return special.kv(1/3,p)
        return integrate.quad(integrand1, s, np.inf, epsabs=1e-45, epsrel=1e-10)[0]
    
    part1 = i1(xi)
    part2 = (2 + 1.5 * g * xi) * special.kv(2/3, xi)

    return coeff * (part1 - part2)


# The radiation power spectrum
# x = electron quantum parameter, g = photon quantum parameter
def P_rad_Synch(x, g):
    coeff = -g / (np.pi * np.sqrt(3) * x**2)
    xi = 2 * g / (3*x*(x - g))
    
    def i1(s):
        def integrand1(p):
            return special.kv(1/3,p)
        return integrate.quad(integrand1, s, np.inf, epsabs=1e-45, epsrel=1e-10)[0]
    
    part1 = i1(xi)
    part2 = (2 + 1.5 * g * xi) * special.kv(2/3, xi)
    return coeff * (part1 - part2)


# The spectrum integrated from 0 to some photon quantum parameter y
# if y = x, where x is the electron quantum parameter, this equals the auxiliary function
def N_Synchrotron(x, y):

    coeff = -1 / (np.pi * np.sqrt(3) * x**2)

    def xi(a):
        return (2*a) / (3*x*(x-a))
    
    def i1(s):
        def integrand1(p):
            return special.kv(1/3,p)
        return integrate.quad(integrand1, s, np.inf, epsabs=1e-45, epsrel=1e-10)[0]

    def integrand(s):
        part1 = i1(xi(s))
        part2 = (2 + 1.5 * s * xi(s)) * special.kv(2/3,xi(s))
        return part1 - part2
    
    try:
        integral, error = integrate.quad(integrand, 0, y, epsabs=1e-45, epsrel=1e-10)
    except Exception:
        return None
        
    return coeff * integral


# The exact cumulative probability of synchrotron radiation,
# x = electron quantum parameter, r = ratio
# Here T is the auxiliary function, use the exact value, but calculate it outside this function to avoid 
# calculating it repeatedly when going through different r, as T only depends on chivalue and is time consuming to calculate
def P_SyncExct(x, r, T):
    return N_Synchrotron(x, x * r) / T #/ T_Synchrotron(x)




# Approximation for the auxiliary function
def T_Synch_approx(x):
    G1 = (1 - 403289.4214570542*x*np.exp(-(x + 0.847336033)**2 / 0.0584260036))
    G2 = (1 + 3.26342394*x*np.exp(-(np.log(x) + 7.9555351)**2 / 19.18831074))
    return (5/(2*np.sqrt(3))) * (1 - np.exp(-1.46*(2*np.sqrt(3)/5)*x**(-1/3))) * G1 * G2



# Functions for the a, b, and n parameters needed in the cumulative probability and radiation spectrum approximations, and the solution to the photon quantum parameters
def a_parameter(x):
    return ((1 + 0.1648*np.exp(-2.279*(x - 1.2024)**2))*5.37351138*x**(-0.07620857)/(1 + 1.74094714*x**(1.29054535)) + 0.5369956083824988 + 0.0016994836*(1.4204061722566785 - 0.5369956083824988)*x**(1.77253101) / (1 + 0.00150308200*x**(1.77253101))) * (1 + -0.08494223*np.exp(-4.59296877*(np.log(x) - 1.75725058)**2))

def b_parameter(x):
    return (-1.56564371 + (11.3942951*x**(0.43384444) + 8.79281269*x**(2*0.44067827)) * np.exp(-1.99971294*x**0.46344015) + (-0.38800187*x**2.23683008 + 0.32931219*x**2.35483769) / (1 + 0.04777412*x**2.23683008)) * (0.87786029 + 0.65634725*np.exp(-1.44269312*(np.log(x) + 0.27729158)**2))**0.0661953

def n_parameter(x):
    return ((0.926739*x**(-0.34531647)) / (1 + 1.26885704*x**0.34531647 + 0.30839327*x**1.74193071)) + 0.25000



# Approximation for the cumulative probability
def P_Synch_approx(chivalue, r):
    a = a_parameter(chivalue)
    b = b_parameter(chivalue)
    n = n_parameter(chivalue)
    c = a - b - 1
    numerator = a*(1 - r**(1/3))**(4*n)
    denominator = 1 + b*(1 - r**(1/3))**(2*n) + c*(1 - r**(1/3))**(3*n)
    return 1 - numerator / denominator


# Approximation for the radiation power spectrum: r*T*dp/dr
# Add the exact auxiliary function T that is calculated outside this function as an argument to avoid recalculating it
# when going through different values of r
def Synch_spec_appr(chivalue, r, T):
    a = a_parameter(chivalue)
    b = b_parameter(chivalue)
    n = n_parameter(chivalue)
    c = a - b - 1
    numerator = (4 + 2*b*(1 - r**(1/3))**(2*n) + c*(1 - r**(1/3))**(3*n)) * (1 - r**(1/3))**(4*n - 1)
    denominator = 1 + b*(1 - r**(1/3))**(2*n) + c*(1 - r**(1/3))**(3*n)
    return T*a*n*(1/3)*numerator*r**(1/3) / (denominator**2)


# Ensures that cubic root returns a real number
def curoot(x):
    if x < 0:
        return -(abs(x))**(1/3)
    else:
        return x**(1/3)


# The ratio of quantum parameter r = chi_gamma/chi_electron taken from the initial electron by an emitted photon
# z is the generated random number between 0 and 1
# Multiply by chi_electron to get chi_gamma of the emitted photon
def r_solutions(z, chi):
    a = a_parameter(chi)
    b = b_parameter(chi)
    n = n_parameter(chi)

    c = a - b - 1

    B = -b*(1 - z)
    C = -c*(1 - z)
    D = z - 1

    K = B/a - (3*C**2) /(8*a**2)
    M = (C**3) /(8*a**3) - (B*C)/(2*a**2)
    N = (D/a) + (B*C**2) / (16*a**3) - (3*C**4) / (256*a**4)
  #  print('K is equal to: ', K)

    p = -(12*N + K**2)/(12)
    q = (-2*K**3 - 36*K*N + 108*(K*N - M**2 / 4)) / 216

    t = curoot(-q/2 + np.sqrt((q**2) / 4 + (p**3) / 27)) + curoot(-q/2 - np.sqrt((q**2) / 4 + (p**3) / 27))
    y = t + K/6

    determinant = q**2 / 4 + p**3 / 27 # This needs to be positive to use Gardano's method when solving the qubic
    if determinant < 0:
        print('Negative determinant!')

    if 2*y - K < 0: # The 2y - K needs to be positive to avoid complex solutions
        print('Negative under the root!: ', 2*y - K)
        print('With chivalue: ', chi)
        print('With z-value: ', z)
    root = np.sqrt(2*y - K)

    # Two of the four quartic roots are physical. The sign of the discriminant below picks
    # the one that stays real and positive; at the crossover the quartic has a double root,
    # so both expressions agree there and the switch adds no discontinuity.
    if -2*y - K - 2*M/root >= 0:
        u = (root + np.sqrt(-2*y - K - 2*M/root)) / 2
    else:
        u = (-root + np.sqrt(-2*y - K + 2*M/root)) / 2

    x = u - C/(4*a)

    if x < 0:
        # Round-off has pushed the shifted root below zero; the photon then takes the whole
        # of the electron quantum parameter.
        sol = 1
    elif z < 0.998 or chi >= 0.1:
        sol = (1 - x**(1/n))**3
    else:
        # Classical corner: soft emission with the random number close to one. Here C/(4a)
        # is minute beside u, so forming x = u - C/(4a) and only then raising it to 1/n
        # cancels away most of the significant digits. Expanding (u - C/4a)^(1/n) about u
        # to second order keeps them.
        x = u**(1/n) - u**(1/n - 1)*C/(4*a*n) - (C/(4*a*n))**2 * ((n - 1)*u**(1/n - 2))/2
        sol = (1 - x)**3

    return u, sol # u is the root of the quartic equation that is used, sol is the value of r given by this u


