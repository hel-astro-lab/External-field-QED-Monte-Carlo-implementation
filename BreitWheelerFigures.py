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
import matplotlib as mpl


t_data = pd.read_csv('T_BW_tab090226.csv')
n_data = pd.read_csv('N_BW_tab090226.csv')


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


# 2.003260712099797 = 0.75681529/0.37779171
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


# THE FIGURES

# The auxiliary function of the Breit-Wheeler process

if False:

    chismallvalues = [0.085, 0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.825, 0.875, 0.915, 0.95, 0.98]
    chiaddvalues = np.logspace(np.log10(5050), np.log10(10000), 15)
    Numeric_values = np.concatenate((np.array([T_BW(x) for x in chismallvalues]), t_data['T'], np.array([T_BW(x) for x in chiaddvalues])))
    chivalues = np.concatenate((chismallvalues, t_data['chi'], chiaddvalues))

    T_approx = np.array([T_BW_approx(j) for j in chivalues])



    print(f'Maximum numeric value: {max(Numeric_values)}')
    diff = Numeric_values - T_approx
    diffrel = abs(diff / Numeric_values)


    fig = plt.figure(1, figsize=(4.9*0.88, 6*0.95))

    lz = 10
    fz = 12

    # add ticks to both sides 
    plt.rc('xtick', top = True)
    plt.rc('ytick', right = True)

    plt.rc('font',  family='serif')
    plt.rc('text',  usetex=True)

    # make labels slightly smaller 
    plt.rc('xtick', labelsize=lz)
    plt.rc('ytick', labelsize=lz)
    plt.rc('axes',  labelsize=(lz-2))
    plt.rc('legend',  handlelength=2.0)

    # number of rows and columns for the figure
    nrow_fig = 2
    ncol_fig = 1

    gs = plt.GridSpec(nrow_fig, ncol_fig)
    gs.update(wspace = 0.2)
    gs.update(hspace = 0.0)

    axs = np.empty( (nrow_fig,ncol_fig), dtype=object)

    for j in range(ncol_fig):
        for i in range(nrow_fig):
            axs[i,j] = plt.subplot(gs[i,j])
            axs[i,j].minorticks_on()
            axs[i,j].set_xlabel(r'$\chi_\gamma$', fontsize=fz)
           # axs[i,j].grid(which='major')
            axs[i,j].set_xscale('log')
            axs[i,j].set_yscale('log')

    
    axs[0,0].plot(chivalues, Numeric_values, 
                  color='Black',
                  alpha=1,
                  lw=1.5,
                  label='Numerical',
                  linestyle='solid')
    
    axs[0,0].plot(chivalues, T_approx,
                  color='C1',
                  alpha=1,
                  lw=1.5,
                  label='Approx',
                  linestyle='--')


    axs[1,0].plot(chivalues, diffrel,
                  color='C1',
                  alpha=1,
                  lw=1,
                  label='Alt',
                  linestyle='solid')

    axs[0,0].set_ylabel(r'$T_{\mathrm{BW}}(\chi_\gamma)$', fontsize=fz)
    axs[1,0].set_ylabel(r'$\mathrm{Relative~ error}$', fontsize=fz)
    axs[0,0].set_ylim(1e-11, 1)
    
    axleft    = 0.2
    axbottom  = 0.2
    axright   = 0.94
    axtop     = 0.9

    fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)
    #plt.savefig('AuxTBW180326joinedxjulkaisu.pdf')
    plt.show()


# The cumulative probability and spectrum compared with the approximations with multiple different chi_photon values

if False:
    figstuple = (4.95*1.075, 6.35*1.075)
    lz = 15
    fz = 15
    legendfz = 'medium'
    fig = plt.figure(1, figsize=figstuple)
    chivalues = [1, 10, 100, 1500]
    cmap = mpl.colormaps['plasma']
    colors = cmap(np.linspace(0.0, 0.9, len(chivalues)))

    # add ticks to both sides 
    plt.rc('xtick', top = True)
    plt.rc('ytick', right = True)

    plt.rc('font',  family='serif')
    plt.rc('text',  usetex=True)

    # make labels slightly smaller 
    plt.rc('xtick', labelsize=lz)
    plt.rc('ytick', labelsize=lz)
    plt.rc('axes',  labelsize=lz)
    plt.rc('legend',  handlelength=2.0)

    # number of rows and columns for the figure
    nrow_fig = 2
    ncol_fig = 1

    gs = plt.GridSpec(nrow_fig, ncol_fig)
    gs.update(wspace = 0.2)
    gs.update(hspace = 0.0)

    axs = np.empty( (nrow_fig,ncol_fig), dtype=object)

    for j in range(ncol_fig):
        for i in range(nrow_fig):
            axs[i,j] = plt.subplot(gs[i,j])
            axs[i,j].minorticks_on()
            axs[i,j].set_xlabel(r'$r$', fontsize=fz)

    
    chis = enumerate(chivalues)
    for n, chi_e in chis:
        rvalues = np.linspace(1e-4, 0.9999, 125)
        t = T_BW(chi_e)
        def P_BW(c, r):
            return N_BW(c, c*r) / t
        P_numeric = np.array([P_BW(chi_e, i) for i in rvalues])

        P = []
        for i in rvalues:
            if i >=1/2:
                P.append(P_BW_approx(chi_e, i))
            else:
                P.append(1 - P_BW_approx(chi_e, (1-i)))
        P_trial = np.array(P)
        diffrel_P = abs((P_numeric - P_trial) / P_numeric)
        print(max(P_trial))

        axs[0,0].plot(rvalues, P_numeric,
                    alpha=1,
                    color='C' + str(n+1),
                    lw=1,
                    label=f'$\\chi_\\gamma = {chi_e}$',
                    linestyle='solid')
        
        axs[0,0].plot(rvalues, P_trial,
                    color='Black',
                    alpha=0.9,
                    lw=0.975,
                    linestyle=':')

        
        axs[1,0].plot(rvalues, diffrel_P,
                    alpha=1,
                    lw=1,
                    color = 'C' + str(n+1),
                    label=f'$\\chi_\\gamma = {chi_e}$',
                    linestyle='solid')


        #axs[0,0].legend(fontsize=legendfz, loc='lower right')
        axs[1,0].set_ylabel(r"$\mathrm{Relative~\ error}$", fontsize=fz)
        axs[1,0].set_yscale('log')
        axs[0,0].set_ylabel(r"$p_{\mathrm{BW}}(\chi_\gamma,r)$", fontsize=fz)
        #axs[1,0].legend(fontsize=legendfz, loc='upper right')
        axs[1,0].set_ylim(1e-4, 2e-1)

        
        axleft    = 0.16
        axbottom  = 0.2
        axright   = 0.94
        axtop     = 0.9

        fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)
    

#    plt.savefig('BWprobplot120226joinedxjulkaisu.pdf')
    plt.show()


    fig = plt.figure(1, figsize=figstuple) 

    # add ticks to both sides 
    plt.rc('xtick', top = True)
    plt.rc('ytick', right = True)

    plt.rc('font',  family='serif')
    plt.rc('text',  usetex=True)

    # make labels slightly smaller 
    plt.rc('xtick', labelsize=lz)
    plt.rc('ytick', labelsize=lz)
    plt.rc('axes',  labelsize=lz)
    plt.rc('legend',  handlelength=2.0)

    # number of rows and columns for the figure
    nrow_fig = 2
    ncol_fig = 1

    gs = plt.GridSpec(nrow_fig, ncol_fig)
    gs.update(wspace = 0.2)
    gs.update(hspace = 0.0)

    axs = np.empty( (nrow_fig,ncol_fig), dtype=object)

    for j in range(ncol_fig):
        for i in range(nrow_fig):
            axs[i,j] = plt.subplot(gs[i,j])
            axs[i,j].minorticks_on()
            axs[i,j].set_xlabel(r'$r$', fontsize=fz)


    chis = enumerate(chivalues)
    for n, chi_e in chis:
        t = T_BW(chi_e)
        rvalues = np.concatenate((np.linspace((0.0001), (0.05), 2500), np.linspace(51e-3, 0.9495, 120),np.linspace((0.95), (0.9999), 2500)))
        N_numeric = np.array([N_BW_spectrum(chi_e, i*chi_e) for i in rvalues])
        N = []
        for i in rvalues:
            if i >=1/2:
                N.append(N_BW_approx(chi_e, i, t))
            else:
                N.append(N_BW_approx(chi_e, (1-i), t))
        N_trial = np.array(N)
        diffrel_N = abs((N_numeric - N_trial) / N_numeric)

        
        axs[0,0].plot(rvalues, N_numeric,
                    alpha=1,
                    color='C' + str(n+1),
                    lw=1,
                    label=f'$\\chi_\\gamma = {chi_e}$',
                    linestyle='solid')
        
        axs[0,0].plot(rvalues, N_trial,
                    color='Black',
                    alpha=0.9,
                    lw=0.975,
                    linestyle=':')
        
        axs[1,0].plot(rvalues, diffrel_N,
                    alpha=1,
                    color='C' + str(n+1),
                    lw=1,
                    label=f'$\\chi_\\gamma = {chi_e}$',
                    linestyle='solid')


        axs[0,0].set_ylabel(r"$d^2 N_{\mathrm{BW}}/ d\chi_e dt$", fontsize=fz)
        #axs[0,0].set_yscale('log')
        #axs[0,0].legend(fontsize='x-small', loc='lower left')
        axs[1,0].set_ylabel(r"$\mathrm{Relative~\ error}$", fontsize=fz)
        axs[1,0].set_yscale('log')
        #axs[0,0].set_ylim(1e-6, max(N_trial)*1.75)
        axs[1,0].set_ylim(1e-4,0.9)
        #axs[1,0].legend(fontsize=legendfz)
        axs[1,0].set_ylim(5e-4, 3e0)
        

        fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)
    
#    plt.savefig('BWSpectrumplots120226joinedxjulkaisu.pdf')

    plt.show()


# The approximations for the cumulative probability and spectrum over the (r, chi_photon)-plane
# and corresponding relative and absolute errors compared to the exact numerically calculated functions

if False:

    fig = plt.figure(1, figsize=(6.75*0.8, 7.25*1.2))

    fz = 13

    # add ticks to both sides
    plt.rc('xtick', top = True)
    plt.rc('ytick', right = True)

    plt.rc('font',  family='serif')
    plt.rc('text',  usetex=True)

    # make labels slightly smaller 
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes',  labelsize=13)
    plt.rc('legend',  handlelength=4.0)

    gs0 = plt.GridSpec(2, 1, figure=fig, hspace=0.0165, height_ratios=[2.9, 1])

    # number of rows and columns for the figure
    nrow_fig = 2
    ncol_fig = 2

    gs1 = gs0[0].subgridspec(nrow_fig, ncol_fig, wspace=0.025, hspace=0.45)

    axs1 = np.empty((nrow_fig,ncol_fig), dtype=object)
    for j in range(ncol_fig):
        for i in range(nrow_fig):
            axs1[i,j] = plt.subplot(gs1[i,j])
            axs1[i,j].set_yticks([])
            axs1[i,j].set_xlabel(r'$r$', fontsize=(fz+1))
            axs1[i,j].set_yscale('log')
            if j == 0:
                axs1[i,j].set_ylabel(r'$\chi_\gamma$', fontsize=(fz+1))

    # number of rows and columns for the figure
    error_nrow_fig = 1
    error_ncol_fig = 2

    gs2 = gs0[1].subgridspec(error_nrow_fig, error_ncol_fig, wspace=0.025, hspace=0.05)

    axs2 = np.empty( (error_nrow_fig,error_ncol_fig), dtype=object)
    for j in range(error_ncol_fig):
        for i in range(error_nrow_fig):
            axs2[i,j] = plt.subplot(gs2[i,j])
            axs2[i,j].set_yticks([])
            axs2[i,j].set_xlabel(r'$r$', fontsize=(fz+1))
            axs2[i,j].set_yscale('log')
            if j == 0:
                axs2[i,j].set_ylabel(r'$\chi_\gamma$', fontsize=(fz+1))




    pvalues = n_data['r'].values
    chivalues = t_data['chi'].values
    x = len(chivalues)
    y = len(pvalues)
    N2Dvalues = np.zeros((x, y))
    N2Dapprox = np.zeros((x, y))
    P2Dvalues = np.zeros((x, y))
    P2Dapprox = np.zeros((x, y))
    cbarticklevels = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    cbarabsticklevels = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]


    for i, chi in enumerate(chivalues):
        for j, r in enumerate(pvalues):
            t = t_data['T'].values[i]
            column = 'chi' + str(i)
            probability = n_data[column].values[j] / t
            if probability < 1:
                P2Dvalues[i, j] = probability
            if probability >= 1:
                P2Dvalues[i, j] = 1
            N2Dvalues[i, j] = N_BW_spectrum(chi, r*chi)
            if r >= 1/2:
                N2Dapprox[i, j] = N_BW_approx(chi, r, t)
                P2Dapprox[i, j] = P_BW_approx(chi, r)
            else:
                N2Dapprox[i, j] = N_BW_approx(chi, (1 - r), t)
                P2Dapprox[i, j] = 1 - P_BW_approx(chi, (1 - r))

    
    SpectrumDiff = N2Dvalues - N2Dapprox
    ProbDiff = P2Dvalues - P2Dapprox
    SpectrumRelD = abs(SpectrumDiff / N2Dvalues)
    ProbRelD = abs(ProbDiff / P2Dvalues)
    print(P2Dvalues.max())
    print(P2Dapprox.max())


    axleft    = 0.14
    axbottom  = 0.2
    axright   = 0.94
    axtop     = 0.9


    axs1[0,0] = plt.subplot(gs1[0,0])
    axs1[0,0].minorticks_on()
    axs1[0,0].yaxis.set_tick_params(which='both', left=True, right=False)
    r1, chi1 = np.meshgrid(pvalues, chivalues)
    axs1[0,0] = plt.contourf(r1, chi1, N2Dapprox, cmap=plt.cm.plasma, levels=20)
    contour1 = plt.colorbar(location='top', format='%.2f')
    contour1.set_ticks([4e-2, 8e-2, 1.2e-1])
    contour1.ax.tick_params(labelsize=12)
    contour1.set_label(r'$d^2 N_{BW} / d\chi_e dt$')

    axs1[0,1] = plt.subplot(gs1[0,1])
    axs1[0,1].minorticks_on()
    axs1[0,1].yaxis.set_tick_params(which='both', left=False, right=True)
    r3, chi3 = np.meshgrid(pvalues, chivalues)
    axs1[0,1] = plt.contourf(r3, chi3, P2Dapprox, cmap=plt.cm.cividis, levels=20)
    plt.yticks([])
    contour3 = plt.colorbar(location='top')
    contour3.set_ticks([2.5e-1, 5e-1, 7.5e-1])
    contour3.ax.tick_params(labelsize=12)
    contour3.set_label(r'$p_{\mathrm{BW}}(\chi_\gamma, r)$')


    axs1[1,0] = plt.subplot(gs1[1,0])
    axs1[1,0].minorticks_on()
    axs1[1,0].yaxis.set_tick_params(which='both', left=True, right=False)
    r2, chi2 = np.meshgrid(pvalues, chivalues)
    axs1[1,0].xaxis.set_tick_params(which='both', bottom=False, top=True)
    axs1[1,0] = plt.contourf(r2, chi2, SpectrumRelD, cmap=plt.cm.coolwarm, norm=col.LogNorm(), levels=np.array([1e-3, 3e-3,1e-2,3e-2,1e-1,3e-1]), extend='both')
    contour2 = plt.colorbar(location='top')
    contour2.set_label(r'$\mathrm{Rel./abs.~ errors}$')


    axs1[1,1] = plt.subplot(gs1[1,1])
    axs1[1,1].minorticks_on()
    axs1[1,1].yaxis.set_tick_params(which='both', left=False, right=True)
    axs1[1,1].xaxis.set_tick_params(which='both', bottom=False, top=True)
    r4, chi4 = np.meshgrid(pvalues, chivalues)
    plt.yticks([])
    axs1[1,1] = plt.contourf(r4, chi4, ProbRelD, cmap=plt.cm.coolwarm, norm=col.LogNorm(), levels=np.array(cbarticklevels), extend='both')
    contour4 = plt.colorbar(location='top')
    contour4.set_label(r'$\mathrm{Rel./abs.~ errors}$')



    axs2[0,0] = plt.subplot(gs2[0,0])
    axs2[0,0].minorticks_on()
    axs2[0,0].yaxis.set_tick_params(which='both', left=True, right=False)
    r2, chi2 = np.meshgrid(pvalues, chivalues)
    axs2[0,0].xaxis.set_tick_params(which='both', bottom=True, top=False)
    axs2[0,0] = plt.contourf(r2, chi2, np.abs(SpectrumDiff), cmap=plt.cm.coolwarm, norm=col.LogNorm(), levels=np.array(cbarabsticklevels), extend='both')

    axs2[0,1] = plt.subplot(gs2[0,1])
    axs2[0,1].minorticks_on()
    axs2[0,1].yaxis.set_tick_params(which='both', left=False, right=True)
    axs2[0,1].xaxis.set_tick_params(which='both', bottom=True, top=False)
    r4, chi4 = np.meshgrid(pvalues, chivalues)
    plt.yticks([])
    axs2[0,1] = plt.contourf(r4, chi4, np.abs(ProbDiff), cmap=plt.cm.coolwarm, norm=col.LogNorm(), levels=np.array(cbarabsticklevels), extend='both')

    fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)
    #plt.savefig('BW2Dplots.11.2.26julkaisu.pdf')
    plt.show()


# The chi_electron values solved with the inverted approximation of cumulative probability,
# and a comparison to the numerical solutions (here, just the cumulative probability plotted with the axis inverted)

if False:

    epsilon = 0.0001
    zvalues = np.linspace(epsilon, 1 - epsilon, 500)
    fig = plt.figure(1, figsize=(4.95*1.1, 6.35*1.2))

    # add ticks to both sides
    plt.rc('xtick', top = True)
    plt.rc('ytick', right = True)

    plt.rc('font',  family='serif')
    plt.rc('text',  usetex=True)

    # make labels slightly smaller 
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('axes',  labelsize=14)
    plt.rc('legend',  handlelength=4.0)

    # number of rows and columns for the figure
    nrow_fig = 2
    ncol_fig = 1

    gs = plt.GridSpec(nrow_fig, ncol_fig)
    gs.update(wspace = 0.3)
    gs.update(hspace = 0.0)

    axs = np.empty( (nrow_fig,ncol_fig), dtype=object)

    for j in range(ncol_fig):
        for i in range(nrow_fig):
            axs[i,j] = plt.subplot(gs[i,j])
            axs[i,j].minorticks_on()
        
    chievalues = np.zeros(len(zvalues))
    Papproxvalues = np.zeros(len(zvalues))
    cmap = plt.cm.plasma
    ccmap = plt.cm.binary
    chigamma = [1, 10, 100, 1500]
    colors = cmap(np.linspace(0.0, 0.9, len(chigamma)))
    ccolors = ['orange', 'green', 'red', 'black']
    cbarticklevels = [1e-1, 3e-1, 1, 3, 10, 30, 100, 300, 1000, 3000]




    for j, chi in enumerate(chigamma):
        t = T_BW(chi)
        for i, z in enumerate(zvalues):
            column = 'chi' + str(j)
            probability = N_BW(chi, chi*z) / t
            if z >= 1/2:
                Papproxvalues[i] = probability
            else:
                Papproxvalues[i] = probability
        
        c = colors[j]

        axs[1,0].plot(Papproxvalues, zvalues,
                    color='C' + str(j + 1),
                    alpha=1,
                    lw=1,
                    linestyle='solid')


    for j, chi in enumerate(chigamma):
        for i, z in enumerate(zvalues):
            column = 'chi' + str(j)
            if z >= 1/2:
                chievalues[i] = r_electron(z, chi)
            else:
                chievalues[i] = (1 - r_electron(1 - z, chi))

        axs[1,0].plot(zvalues, chievalues,
                    color= 'Black',
                    alpha = 1.0,
                    lw = 1.0,
                    label=f"$\chi_\gamma=~${chi}",
                    linestyle=':',
                    )
        



        axs[1,0].minorticks_on()
        axs[1,0].set_xlabel(r"$r=\chi_e/\chi_\gamma$", fontsize=15)
        axs[1,0].grid(which='major')
        #axs[1,0].legend(fontsize='medium')


    p1 = np.log10(0.01)
    p2 = np.log10(0.999)
    c1 = np.log10(1)
    c2 = np.log10(5000)
    no = 250
    zvalues = np.linspace(0.0005, 0.9995, no)
    chivalues = np.logspace(c1, c2, no)
    x = len(chivalues)
    y = len(zvalues)
    chie2Dvalues = np.zeros((x, y))

    for i, chi in enumerate(chivalues):
        for j, z in enumerate(zvalues):
            if z >= 1/2:
                chie2Dvalues[i, j] = r_electron(z, chi)*chi
            else:
                chie2Dvalues[i, j] = (1 - r_electron(1 - z, chi))*chi


    axs[0,0] = plt.subplot(gs[0,0])
    axs[0,0].minorticks_on()
    axs[0,0].set_yscale('log')
    axs[0,0].set_xlabel(r"$\zeta$", fontsize=15)
    axs[1,0].set_xlabel(r"$\zeta$", fontsize=15)
    axs[0,0].set_ylabel(r"$\chi_{\gamma}$", fontsize=15)
    axs[1,0].set_ylabel(r"$r(\chi_{\gamma}, \zeta)$", fontsize=15)
    axs[1,0].set_ylim(0,1.1)
    
    r, chi = np.meshgrid(zvalues, chivalues)
    axs[0,0] = plt.contourf(r, chi, chie2Dvalues, cmap=plt.cm.plasma, norm=col.LogNorm(), levels = cbarticklevels, extend='both')
    contour = plt.colorbar(location='top')
    contour.set_label(r'$\chi_e(\chi_{\gamma}, \zeta)$', fontsize=15)
    contour.set_ticks(cbarticklevels)
    contour.set_ticklabels([0.1, '', 1, '', 10, '', 100, '', 1000, ''])


    axleft    = 0.14
    axbottom  = 0.2
    axright   = 0.94
    axtop     = 0.9

    fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)

#    fig.savefig('BWQuarticPadeElectonChi.6.4.26julkaisu.pdf')

    plt.show()


# The obtained optimal values of the auxiliary parameters compared to their approximating forms

if False:
    for variable in ['Pn', 'Pa', 'Pb']:
        figz = (5.5, 3.75)
        fig = plt.figure(1, figsize=figz)
        fz = 18
        lz = 16

        if variable == 'Pa':
            data = pd.read_csv('BWHO{4;0,1,2}PadeparametersOnlyAandB100526.csv')
            plotcolor = 'C0'
            def model(x):
                return a_parameter(x)
        elif variable == 'Pb':
            data = pd.read_csv('BWHO{4;0,1,2}PadeparametersOnlyAandB100526.csv')
            plotcolor = 'C1'
            def model(x):
                return b_parameter(x)
        elif variable == 'Pn':
            data = pd.read_csv('BWHO{4;0,1,2}PadeparamALL100526.csv')
            plotcolor = 'C2'
            def model(x):
                return n_parameter(x)
        Numeric_values = data[variable].values


        # add ticks to both sides 
        plt.rc('xtick', top = True)
        plt.rc('ytick', right = True)

        plt.rc('font',  family='serif')
        plt.rc('text',  usetex=True)

        # make labels slightly smaller
        plt.rc('xtick', labelsize=lz)
        plt.rc('ytick', labelsize=lz)
        plt.rc('axes',  labelsize=lz)
        plt.rc('legend',  handlelength=2.0)

        # number of rows and columns for the figure
        nrow_fig = 1
        ncol_fig = 1

        gs = plt.GridSpec(nrow_fig, ncol_fig)
        gs.update(wspace = 0.4)
        gs.update(hspace = 0.4)

        axs = np.empty( (nrow_fig,ncol_fig), dtype=object)

        for j in range(ncol_fig):
            for i in range(nrow_fig):
                axs[i,j] = plt.subplot(gs[i,j])
                axs[i,j].minorticks_on()
                axs[i,j].set_xscale('log')
                axs[i,j].set_xlabel(r"$\chi_\gamma$", fontsize=fz)


        chi = data['Chi'].values
        xvalues = chi
        model_values = [model(i) for i in xvalues]
        print(min(model_values))
        error = Numeric_values - model_values


        axs[0,0].plot(xvalues, Numeric_values,
                    color=plotcolor,
                    alpha = 1.0,
                    lw = 1.0,
                    linestyle='solid'
                    )


        
        axs[0,0].plot(xvalues, model_values,
                    color='Black',
                    alpha = 1,
                    lw = 1.0,
                    linestyle=':'
                    )


        if variable == 'Pa':
            axs[0,0].set_title(r'$a(\chi_\gamma)$', fontsize=(fz+1))
        if variable == 'Pb':
            axs[0,0].set_title(r'$b(\chi_\gamma)$', fontsize=(fz+1))
        if variable == 'Pn':
            axs[0,0].set_title(r'$n(\chi_\gamma)$', fontsize=(fz+1))



        axleft    = 0.14
        axbottom  = 0.2
        axright   = 0.94
        axtop     = 0.9

        fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)

        name = 'QuarticBW' + variable + 'param110526julkaisu.pdf'
        plt.savefig(name)
        plt.show()

    # The relative error
        fig = plt.figure(1, figsize=figz)


        # add ticks to both sides 
        plt.rc('xtick', top = True)
        plt.rc('ytick', right = True)

        plt.rc('font',  family='serif')
        plt.rc('text',  usetex=True)

        # make labels slightly smaller
        plt.rc('xtick', labelsize=lz)
        plt.rc('ytick', labelsize=lz)
        plt.rc('axes',  labelsize=lz)
        plt.rc('legend',  handlelength=2.0)

        # number of rows and columns for the figure
        nrow_fig = 1
        ncol_fig = 1

        gs = plt.GridSpec(nrow_fig, ncol_fig)
        gs.update(wspace = 0.4)
        gs.update(hspace = 0.4)

        axs = np.empty( (nrow_fig,ncol_fig), dtype=object)

        for j in range(ncol_fig):
            for i in range(nrow_fig):
                axs[i,j] = plt.subplot(gs[i,j])
                axs[i,j].minorticks_on()
                axs[i,j].set_xscale('log')
                axs[i,j].set_ylim(5e-4,2e-1)


        axs[0,0].plot(chi, abs(error/Numeric_values),
                    color='magenta',
                    alpha = 1.0,
                    lw = 1.0,
                    linestyle='solid'
                    )


        for j in range(ncol_fig):
            for i in range(nrow_fig):
                axs[i,j].set_xscale('log')
                axs[i,j].set_xlabel(r"$\chi_\gamma$", fontsize=fz)
                axs[i,j].set_yscale('log')

        axs[0,0].set_title(r'$\mathrm{Relative~ error}$', fontsize=16)

        name = 'QuarticBW' + variable + 'paramrelerror110526julkaisu.pdf'
        


        axleft    = 0.14
        axbottom  = 0.2
        axright   = 0.94
        axtop     = 0.9

        fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)
        plt.savefig(name)
        plt.show()

    # The absolute error
        fig = plt.figure(1, figsize=figz)


        # add ticks to both sides 
        plt.rc('xtick', top = True)
        plt.rc('ytick', right = True)

        plt.rc('font',  family='serif')
        plt.rc('text',  usetex=True)

        # make labels slightly smaller
        plt.rc('xtick', labelsize=lz)
        plt.rc('ytick', labelsize=lz)
        plt.rc('axes',  labelsize=lz)
        plt.rc('legend',  handlelength=2.0)

        # number of rows and columns for the figure
        nrow_fig = 1
        ncol_fig = 1

        gs = plt.GridSpec(nrow_fig, ncol_fig)
        gs.update(wspace = 0.4)
        gs.update(hspace = 0.4)

        axs = np.empty( (nrow_fig,ncol_fig), dtype=object)

        for j in range(ncol_fig):
            for i in range(nrow_fig):
                axs[i,j] = plt.subplot(gs[i,j])
                axs[i,j].minorticks_on()



        axs[0,0].plot(chi, abs(error),
                    color='limegreen',
                    alpha = 1.0,
                    lw = 1.0,
                    linestyle='solid'
                    )


        for j in range(ncol_fig):
            for i in range(nrow_fig):
                axs[i,j].minorticks_on()
                axs[i,j].set_xlabel(r"$\chi_\gamma$", fontsize=fz)
                axs[i,j].set_xscale('log')
                axs[i,j].set_yscale('log')

        axs[0,0].set_title(r'$\mathrm{Absolute~ error}$', fontsize=16)


        axleft    = 0.14
        axbottom  = 0.2
        axright   = 0.94
        axtop     = 0.9

        fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)

        name = 'QuarticBW' + variable + 'paramabserror110526julkaisu.pdf'
        plt.savefig(name)

        plt.show()


# The values of the auxiliary parameters gathered from the probability (Pa, Pb, Pn) and the radiation power spectrum (Na, Nb, Nn)

if False:

    data = pd.read_csv('BWHO{4;0,1,2}PadeparamALL100526.csv')

    fig = plt.figure(1, figsize=(8.5, 2.8))

    # add ticks to both sides 
    plt.rc('xtick', top = True)
    plt.rc('ytick', right = True)

    plt.rc('font',  family='serif')
    plt.rc('text',  usetex=True)

    # make labels slightly smaller
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes',  labelsize=17)
    plt.rc('legend',  handlelength=2.0)

    # number of rows and columns for the figure
    nrow_fig = 1
    ncol_fig = 3

    gs = plt.GridSpec(nrow_fig, ncol_fig)
    gs.update(wspace = 0.25)
    gs.update(hspace = 0.4)

    axs = np.empty( (nrow_fig,ncol_fig), dtype=object)

    for j in range(ncol_fig):
        for i in range(nrow_fig):
            axs[i,j] = plt.subplot(gs[i,j])
            axs[i,j].minorticks_on()
            axs[i,j].set_xscale('log')
            axs[i,j].set_xlabel(r"$\chi_\gamma$", fontsize=18)
            axs[i,j].set_xticks([1, 1e1, 1e2, 1e3])


    chi = data['Chi'].values

    aP = data['Pa'].values
    bP = data['Pb'].values
    nP = data['Pn'].values
    aN = data['Na'].values
    bN = data['Nb'].values
    nN = data['Nn'].values


    if True:
        axs[0,0].plot(chi, aP, 
                    color='C0',
                    alpha = 1.0,
                    lw = 1.0,
                    label=r'$\mathrm{Spectrum}$',
                    linestyle='solid'
                    )


    if True:
        axs[0,0].plot(chi, aN,
                    color='Black',
                    alpha = 1,
                    lw = 1.0,
                    label=r"$P_{S}$",
                    linestyle=':'
                    )


    axs[0,1].plot(chi, bP,
                color='C1',
                alpha = 1.0,
                lw = 1.0,
                label=r'$P_{S}$',
                linestyle='solid'
                )


    axs[0,1].plot(chi, bN,
                color='Black',
                alpha = 1,
                lw = 1.0,
                label=r'$\mathrm{Spectrum}$',
                linestyle=':'
                )



    axs[0,2].plot(chi, nP,
                color='C2',
                alpha = 1.0,
                lw = 1.0,
                label=r'$P_{S}$',
                linestyle='solid'
                )
    
    axs[0,2].plot(chi, nN,
                color='Black',
                alpha = 1,
                lw = 1.0,
                label=r'$\mathrm{Spectrum}$',
                linestyle=':'
                )



    axs[0,0].set_title(r'$a(\chi_\gamma)$', fontsize=18)
    axs[0,1].set_title(r'$b(\chi_\gamma)$', fontsize=18)
    axs[0,2].set_title(r'$n(\chi_\gamma)$', fontsize=18)


    axleft    = 0.14
    axbottom  = 0.2
    axright   = 0.94
    axtop     = 0.9

    fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)

    #fig.savefig('BWABNparameters110526julkaisu.pdf')
    plt.show()


# Using the derived expression for the inverted cumulative probability 
# to plot the distribution of r-values of the produced electrons

if False:

    fig = plt.figure(1, figsize=(7, 5.5))
    # add ticks to both sides
    plt.rc('xtick', top = True)
    plt.rc('ytick', right = True)

    plt.rc('font',  family='serif')
    plt.rc('text',  usetex=True)

    # make labels slightly smaller 
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('axes',  labelsize=12)
    plt.rc('legend',  handlelength=4.0)

    # number of rows and columns for the figure
    nrow_fig = 2
    ncol_fig = 2

    gs = plt.GridSpec(nrow_fig, ncol_fig)
    gs.update(wspace = 0.2)
    gs.update(hspace = 0.6)

    axs = np.empty( (nrow_fig,ncol_fig), dtype=object)
    epsilon = 0.0001#0.000005

    for j in range(ncol_fig):
        for i in range(nrow_fig):
            axs[i,j] = plt.subplot(gs[i,j])
            axs[i,j].minorticks_on()
            axs[i,j].set_xlabel('$\chi_e/\chi_\gamma$', fontsize=14)
            #axs[i,j].set_yscale('log')


    for n, chi_in in enumerate([1, 10, 100, 1500]):
        if n == 0:
            row = 0
            column = 0
            r_values = np.linspace(epsilon, 1 - epsilon, 300)
        if n == 1:
            row = 0
            column = 1
            r_values = np.linspace(epsilon, 1 - epsilon, 300)
        if n == 2:
            row = 1
            column = 0
            r_values = np.linspace(epsilon, 1 - epsilon, 300)
        if n == 3:
            row = 1
            column = 1
            r_values = np.linspace(epsilon, 1 - epsilon, 300)
        


        N = 300**2
        Np = int(np.sqrt(N))
        chi_outs = np.zeros(N)

        for i in range(N):
            zeta = np.random.uniform(low=epsilon, high=(1 - epsilon))
            if zeta > 0.5:
                chi_outs[i] = r_electron(zeta, chi_in)
            else:
                chi_outs[i] = 1 - r_electron(1 - zeta, chi_in)


        T = T_BW(chi_in)
        def P_BW(chi, r):
            return N_BW(chi, chi*r)/T

        N_spectrum = np.array([N_BW_spectrum(chi_in, chi_in*r) for r in r_values])

        
        dr = r_values[1] - r_values[0]

        #p_values = np.diff(np.array([P_BW(chi_in, r) for r in r_values]))
        r_avg = (r_values[1:] + r_values[:-1])/2
        d_avg = r_avg[1]-r_avg[0]
        H, bins_out = np.histogram(chi_outs, bins=int(np.sqrt(N)+1), density=True)
        bins_center = (bins_out[1:] + bins_out[:-1]) / 2
        N_numeric = N_spectrum / (np.sum(N_spectrum)*dr)
        print('Current chivalue: ', chi_in)



        #axs[row, column].plot(r_avg, p_values/np.sum(p_values*d_avg), color='black')
        axs[row, column].plot(r_values, N_numeric, color='black')
        axs[row, column].set_title(f'$\\chi_\gamma = {chi_in}$', fontsize=14)
        axs[row, column].hist(chi_outs, bins = int(np.sqrt(N)), histtype='step', color='C1', density=True)
    axleft    = 0.14
    axbottom  = 0.2
    axright   = 0.94
    axtop     = 0.9

    fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)
    #plt.savefig('BWProducedSpectrajulkaisu.pdf')
    plt.show()





