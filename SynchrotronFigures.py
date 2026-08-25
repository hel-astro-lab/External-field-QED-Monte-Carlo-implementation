"""Figures for the synchrotron sections of the manuscript.

The physics itself lives in Implement_Synch.py; this script only plots it.
Run from the directory holding this file: it reads the tabulated values from Tables/
and writes the figures to figures/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
import scipy.integrate as integrate
import scipy.special as special

from Implement_Synch import (
    T_Synchrotron, N_Synch_spectrum, P_rad_Synch, N_Synchrotron, P_SyncExct,
    T_Synch_approx, a_parameter, b_parameter, n_parameter,
    P_Synch_approx, Synch_spec_appr, curoot, r_solutions,
)

TABDIR = 'Tables/'
FIGDIR = 'figures/'
os.makedirs(FIGDIR, exist_ok=True)

# The inversion diverges at zeta -> 1, so the random number is drawn from
# (Z_CLIP, 1 - Z_CLIP).
Z_CLIP = 0.0005


# THE FIGURES

# The auxiliary function of synchrotron radiation
if False:

    t_data = pd.read_csv(TABDIR + 'T_Synch_tab030226.csv')
    n_data = pd.read_csv(TABDIR + 'N_Synch_tab030226.csv')

    chiaddvalues = np.logspace(np.log10(125), np.log10(10000), 45)
    Numeric_values = np.concatenate((t_data['T'], np.array([T_Synchrotron(x) for x in chiaddvalues])))
    chivalues = np.concatenate((t_data['chi'], chiaddvalues))

    T_approx = np.array([T_Synch_approx(j) for j in chivalues])


    diff = Numeric_values - T_approx
    diffrel = abs(diff / Numeric_values)

    
    prop = 1
    fig = plt.figure(1, figsize=(4.8*prop, 5.2*prop))

    # add ticks to both sides 
    plt.rc('xtick', top = True)
    plt.rc('ytick', right = True)

    plt.rc('font',  family='serif')
    plt.rc('text',  usetex=True)

    axp = 11
    # make labels slightly smaller
    plt.rc('xtick', labelsize=axp)
    plt.rc('ytick', labelsize=axp)
    plt.rc('axes',  labelsize=axp)
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
            #axs[i,j].set_xlabel(r'$\chi_e$', fontsize='medium')
           # axs[i,j].grid(which='major')
            axs[i, j].set_xlim(1e-4, 1e4)
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
                  label='Approx',
                  linestyle='solid')



    fs = 14
    axs[0,0].get_xaxis().set_visible(False)
    axs[0,0].set_ylabel(r'$T_{\mathrm{S}}(\chi_e)$', fontsize=fs)
    axs[1,0].set_ylabel(r'$\mathrm{Relative~ error}$', fontsize=fs)
    axs[1,0].set_ylim(1e-4,5e-2)
    axs[1,0].set_xlabel(r'$\chi_e$', fontsize=fs)
   # axs[0,0].legend(fontsize='small')
   # for j in range(ncol_fig):
    #    for i in range(nrow_fig):
     #       axs[i,j].legend(fontsize='small')
    
    axleft    = 0.17
    axbottom  = 0.2
    axright   = 0.94
    axtop     = 0.9

    fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)

    #plt.savefig(FIGDIR + 'AuxTSynch020426joinedxaxisjulkaisu.pdf')

    plt.show()


# The cumulative probability and spectrum compared with the approximations with multiple different chi_electron values


if False:
    figstuple = (4.95*1.2, 6.35*1.2)
    lz = 15
    fz = 15
    legendfz = 'medium'
    fig = plt.figure(1, figsize=figstuple)
    cmap = mpl.colormaps['plasma']
    chivalues = [0.0005, 0.05]#, 0.5, 50]

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
           # axs[i,j].grid(which='major')

    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0.0, 0.9, len(chivalues)))
    
    chis = enumerate(chivalues)
    for n, chi_e in chis:
        rvalues = np.concatenate((np.logspace(-5, np.log10(0.5), 75), np.linspace(0.51, 0.95, 55), np.logspace(np.log10(0.955), np.log10(0.9995), 25)))
        t = T_Synchrotron(chi_e)
        def P_SyncExct(c, r):
            return N_Synchrotron(c, c*r) / t
        P_numeric = np.array([P_SyncExct(chi_e, i) for i in rvalues])

        P_trial = np.array([P_Synch_approx(chi_e, i) for i in rvalues])
        diffrel_P = abs((P_numeric - P_trial) / P_numeric)
        c = colors[n]
        print(max(P_trial))

        axs[0,0].plot(rvalues, P_numeric,
                    alpha=1,
                    color='C' + str(n+1), # c,
                    lw=1,
                    label=f'$\\chi_e = {chi_e}$',
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
                    label=f'$\\chi_e = {chi_e}$',
                    linestyle='solid')

        #axs[0,0].legend(fontsize=legendfz, loc='lower right')
        axs[1,0].set_ylabel(r"$\mathrm{Relative~\ error}$", fontsize=fz)
        axs[1,0].set_yscale('log')
        axs[0,0].set_ylabel(r"$p_{\mathrm{S}}(\chi_e,r)$", fontsize=fz)
        #axs[1,0].legend(fontsize=legendfz, loc='upper right')
        axs[1,0].set_ylim(1e-4, 2e-1)
        
        axs[0,0].set_xscale('log')
        #axs[0,0].set_yscale('log')
        axs[1,0].set_xscale('log')
        
        axleft    = 0.17
        axbottom  = 0.2
        axright   = 0.94
        axtop     = 0.9

        fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)
    
#    plt.savefig(FIGDIR + 'syncprobplot040126joinedxjulkaisu.pdf')

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
            #axs[i,j].grid(which='major')

    chis = enumerate(chivalues)
    for n, chi_e in chis:
        t = T_Synchrotron(chi_e)
        N_numeric = np.array([P_rad_Synch(chi_e, i*chi_e) for i in rvalues])
        N_trial = np.array([Synch_spec_appr(chi_e, i, t) for i in rvalues])
        c = colors[n]
        diffrel_N = abs((N_numeric - N_trial) / N_numeric)

        
        axs[0,0].plot(rvalues, N_numeric,
                    alpha=1,
                    color='C' + str(n+1),
                    lw=1,
                    label=f'$\\chi_e = {chi_e}$',
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
                    label=f'$\\chi_e = {chi_e}$',
                    linestyle='solid')

        axs[0,0].set_ylabel(r"$d P_{\mathrm{rad}}/ d\chi_\gamma$", fontsize=fz)
        #axs[0,0].set_yscale('log')
        #axs[0,0].legend(fontsize=legendfz, loc='lower left')
        axs[1,0].set_ylabel(r"$\mathrm{Relative~\ error}$", fontsize=fz)
        axs[1,0].set_yscale('log')
        #axs[0,0].set_ylim(1e-6, max(N_trial)*1.75)
        axs[1,0].set_ylim(1e-4,5e-1)
        #axs[1,0].legend(fontsize=legendfz)
        axs[0,0].set_xscale('log')

        axs[1,0].set_xscale('log')
        axs[1,0].set_ylim(5e-4, 9e-1)
        
       # axleft    = 0.17
       # axbottom  = 0.2
       # axright   = 0.94
       # axtop     = 0.9

        fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)
    
   # plt.savefig(FIGDIR + 'syncSpectrumplots040226joinedxjulkaisu.pdf')

    plt.show()


# The approximations for the cumulative probability and spectrum over the (r, chi_electron)-plane
# and corresponding relative and absolute errors compared to the exact numerically calculated functions
if False:
    t_data = pd.read_csv(TABDIR + 'T_Synch_tab030226.csv')
    n_data = pd.read_csv(TABDIR + 'N_Synch_tab030226.csv')

    fig = plt.figure(1, figsize=(6.34, 10.3))
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

    gs0 = plt.GridSpec(2, 1, figure=fig, hspace=0.0165, height_ratios=[2.85, 1])

    # number of rows and columns for the figure
    nrow_fig = 2
    ncol_fig = 2

    gs1 = gs0[0].subgridspec(nrow_fig, ncol_fig, wspace=0.025, hspace=0.315)


    axs1 = np.empty((nrow_fig,ncol_fig), dtype=object)
    ticklevels = [1e-5, 1e-3, 1e-1]
    for j in range(ncol_fig):
        for i in range(nrow_fig):
            axs1[i,j] = plt.subplot(gs1[i,j])
            #axs1[i,j].minorticks_on()
            axs1[i,j].set_yticks([])
            axs1[i,j].set_xlabel(r'$r$', fontsize=(fz+1))
            axs1[i,j].set_xscale('log')
            axs1[i,j].set_yscale('log')
            axs1[i,j].set_ylim(1e-4,8e1)
            if j == 0:
                axs1[i,j].set_ylabel(r'$\chi_e$', fontsize=(fz+1))

    # number of rows and columns for the figure
    error_nrow_fig = 1
    error_ncol_fig = 2

    gs2 = gs0[1].subgridspec(error_nrow_fig, error_ncol_fig, wspace=0.025, hspace=0.05)


    axs2 = np.empty( (error_nrow_fig,error_ncol_fig), dtype=object)
    for j in range(error_ncol_fig):
        for i in range(error_nrow_fig):
            axs2[i,j] = plt.subplot(gs2[i,j])
            #axs2[i,j].minorticks_on()
            axs2[i,j].set_yticks([])
            axs2[i,j].set_xlabel(r'$r$', fontsize=(fz+1))
            axs2[i,j].set_xscale('log')
            axs2[i,j].set_yscale('log')
            axs2[i,j].set_ylim(1e-4,8e1)
            if j == 0:
                axs2[i,j].set_ylabel(r'$\chi_e$', fontsize=(fz+1))

    cbarticklevels = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]



    pvalues = n_data['r'].values
    chivalues = t_data['chi'].values
    x = len(chivalues)
    y = len(pvalues)
    N2Dvalues = np.zeros((x, y))
    N2Dapprox = np.zeros((x, y))
    P2Dvalues = np.zeros((x, y))
    P2Dapprox = np.zeros((x, y))

    for i, chi in enumerate(chivalues):
        for j, r in enumerate(pvalues):
            t = t_data['T'].values[i]
            column = 'chi' + str(i)
            probability = n_data[column].values[j] / t
            if probability < 1:
                P2Dvalues[i, j] = probability
            if probability >= 1:
                P2Dvalues[i, j] = 1
            N2Dvalues[i, j] = P_rad_Synch(chi, r*chi)
            N2Dapprox[i, j] = Synch_spec_appr(chi, r, t)
            P2Dapprox[i, j] = P_Synch_approx(chi, r)

    
    SpectrumDiff = np.abs(N2Dvalues - N2Dapprox)
    ProbDiff = np.abs(P2Dvalues - P2Dapprox)
    ProbDiff = np.clip(ProbDiff, 1e-6, 10) # Minimum value set

    SpectrumRelD = np.abs(SpectrumDiff / N2Dvalues)
    SpectrumRelD = np.clip(SpectrumRelD, 1e-6, 10)

    ProbRelD = np.abs(ProbDiff / P2Dvalues)
    ProbRelD = np.clip(ProbRelD, 1e-6, 1) # Minimum value set


    axs1[0,0] = plt.subplot(gs1[0,0])
    axs1[0,0].minorticks_on()
    axs1[0,0].yaxis.set_tick_params(which='both', left=True, right=False)
    r1, chi1 = np.meshgrid(pvalues, chivalues)
    axs1[0,0] = plt.contourf(r1, chi1, N2Dapprox, cmap=plt.cm.plasma, levels=20)
    contour1 = plt.colorbar(location='top', format='%.2f')
    contour1.set_ticks([0.06, 0.12, 0.18, 0.24])
    contour1.set_label(r'$dP_{\mathrm{rad}} / d\chi_\gamma$', fontsize=fz)

    axs1[0,1] = plt.subplot(gs1[0,1])
    axs1[0,1].minorticks_on()
    axs1[0,1].yaxis.set_tick_params(which='both', left=False, right=True)
    r3, chi3 = np.meshgrid(pvalues, chivalues)
    plt.yticks([])
    axs1[0,1] = plt.contourf(r3, chi3, P2Dapprox, cmap=plt.cm.cividis, levels=20)
    contour3 = plt.colorbar(location='top', format='%.2f')
    contour3.set_ticks([2.5e-1, 5e-1, 7.5e-1])
    contour3.set_label(r'$p_{\mathrm{S}}(\chi_e, r)$', fontsize=fz)

    axs1[1,0] = plt.subplot(gs1[1,0])
    axs1[1,0].minorticks_on()
    axs1[1,0].xaxis.set_tick_params(which='both', bottom=False, top=True)
    axs1[1,0].yaxis.set_tick_params(which='both', left=True, right=False)
    r2, chi2 = np.meshgrid(pvalues, chivalues)
    plt.xticks([])
    axs1[1,0] = plt.contourf(r2, chi2, SpectrumRelD, cmap=plt.cm.coolwarm, norm=col.LogNorm(), levels=np.array(cbarticklevels), extend='both')
    contour2 = plt.colorbar(location='top')
    contour2.set_label(r'$\mathrm{Rel./abs.~ errors}$', fontsize=fz)


    axs1[1,1] = plt.subplot(gs1[1,1])
    axs1[1,1].minorticks_on()
    axs1[1,1].yaxis.set_tick_params(which='both', left=False, right=True)
    axs1[1,1].xaxis.set_tick_params(which='both', bottom=False, top=True)
    r4, chi4 = np.meshgrid(pvalues, chivalues)
    plt.xticks([])
    plt.yticks([])
    axs1[1,1] = plt.contourf(r4, chi4, ProbRelD, cmap=plt.cm.coolwarm, norm=col.LogNorm(), levels=np.array(cbarticklevels), extend='both')
    contour4 = plt.colorbar(location='top')
    contour4.set_label(r'$\mathrm{Rel./abs.~ errors}$', fontsize=fz)


    axs2[0,0] = plt.subplot(gs2[0,0])
    axs2[0,0].minorticks_on()
    axs2[0,0].xaxis.set_tick_params(which='both', bottom=True, top=False)
    axs2[0,0].yaxis.set_tick_params(which='both', left=True, right=False)
    r2, chi2 = np.meshgrid(pvalues, chivalues)
    axs2[0,0] = plt.contourf(r2, chi2, np.abs(SpectrumDiff), cmap=plt.cm.coolwarm, norm=col.LogNorm(), levels=np.array([3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]), extend='both')


    axs2[0,1] = plt.subplot(gs2[0,1])
    axs2[0,1].minorticks_on()
    axs2[0,1].yaxis.set_tick_params(which='both', left=False, right=True)
    axs2[0,1].xaxis.set_tick_params(which='both', bottom=True, top=False)
    r4, chi4 = np.meshgrid(pvalues, chivalues)
    plt.yticks([])
    axs2[0,1] = plt.contourf(r4, chi4, np.abs(ProbDiff), cmap=plt.cm.coolwarm, norm=col.LogNorm(), levels=np.array([3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]), extend='both')

    axleft    = 0.14
    axbottom  = 0.2
    axright   = 0.94
    axtop     = 0.9

    fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)
    #plt.savefig(FIGDIR + 'Synch2Dplots.5.2.26julkaisu.pdf')
    plt.show()


# The chi_photon values solved with the inverted approximation of cumulative probability,
# and a comparison to the numerical solutions (here, just the cumulative probability plotted with the axes inverted)
if False:

    zvalues = np.concatenate((np.logspace(np.log10(0.0001), np.log10(0.15), 45),np.linspace((0.151), (0.98), 100), np.logspace(np.log10(0.99), np.log10(0.9999), 45))) # 45, 100, 45
    fig = plt.figure(1, figsize=(4.95*1.1, 6.35*1.2))

    fz = 15

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
    gs.update(wspace = 0.25)
    gs.update(hspace = 0.0)

    axs = np.empty( (nrow_fig,ncol_fig), dtype=object)

    for j in range(ncol_fig):
        for i in range(nrow_fig):
            axs[i,j] = plt.subplot(gs[i,j])
            axs[i,j].minorticks_on()

    zpval = np.logspace(-9, np.log10(0.99999), 190)
    chievalues = np.zeros(len(zvalues))
    Papproxvalues = np.zeros(len(zpval))
    cmap = plt.cm.plasma
    ccmap = plt.cm.binary
    chie = [0.001, 0.01, 0.1, 1, 10]
    colors = cmap(np.linspace(0.0, 0.9, len(chie)))
    ccolors = ['orange', 'green', 'red']
    cbarlevels = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10, 30, 100]
    cbarticklevels = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]

    for j, chi in enumerate(chie):
        t = T_Synchrotron(chi)
        def P_SyncExct(c, r):
            return N_Synchrotron(c, c*r) / t
        for i, z in enumerate(zvalues):
            chievalues[i] = r_solutions(z, chi)[1]
        for i, zp in enumerate(zpval):
            Papproxvalues[i] = P_SyncExct(chi, zp)
        
        c = colors[j]

        axs[1,0].plot(Papproxvalues, zpval,
                    color='C' + str(j +1),
                    alpha=1,
                    lw=1,
                    linestyle='solid')

        axs[1,0].plot(zvalues, chievalues,
                    color= 'black',#c,
                    alpha = 1.0,
                    lw = 1.0,
                    label=rf"$\chi_e=~${chi}",
                    linestyle=':',
                    )
        


        axs[1,0].minorticks_on()
        axs[1,0].set_xlabel(r"$r=\chi_\gamma/\chi_e$", fontsize=fz)
      #  axs[0,1].grid(which='major')
        axs[1,0].set_ylim(1e-10,8.5)
        #axs[1,0].legend(fontsize='medium', loc='lower right')


    p1 = np.log10(0.01)
    p2 = np.log10(0.999)
    c1 = np.log10(0.01)
    c2 = np.log10(100)
    no = 500
    zvalues = np.linspace(0.0005, 0.9995, no)
    chivalues = np.logspace(c1, c2, no)
    x = len(chivalues)
    y = len(zvalues)
    chie2Dvalues = np.zeros((x, y))

    print("The evaluation of 2D begins: ")
    for i, chi in enumerate(chivalues):
        for j, z in enumerate(zvalues):
            chie2Dvalues[i, j] = r_solutions(z, chi)[1]*chi


    axs[0,0] = plt.subplot(gs[0,0])
    axs[0,0].minorticks_on()
    axs[0,0].set_yscale('log')
    axs[0,0].set_xlabel(r"$\zeta$", fontsize=fz)
    axs[1,0].set_xlabel(r"$\zeta$", fontsize=fz)
    axs[0,0].set_ylabel(r"$\chi_{e}$", fontsize=fz)
    axs[1,0].set_ylabel(r"$r(\chi_e, \zeta)$", fontsize=fz)
    axs[1,0].set_yscale('log')

    
    
    r, chi = np.meshgrid(zvalues, chivalues)
    axs[0,0] = plt.contourf(r, chi, chie2Dvalues, cmap=plt.cm.plasma, norm=col.LogNorm(), levels = cbarlevels, extend='both')


    contour = plt.colorbar(location='top')
    contour.set_ticks(cbarlevels)
    contour.set_ticklabels(['', 1e-4, '', 1e-3, '', 1e-2, '', 0.1, '', 1, '', 10, '', 100])
    contour.set_label(r'$\chi_\gamma(\chi_e, \zeta)$', fontsize=fz)


    axleft    = 0.17
    axbottom  = 0.2
    axright   = 0.94
    axtop     = 0.9

    fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)

#    fig.savefig(FIGDIR + 'SynchquarticPadePhotonChi.04.03.26julkaisu.pdf')

    plt.show()


# The obtained optimal values of the auxiliary parameters compared to their approximating forms

if False:
    for variable in ['Pn', 'Pa', 'Pb']:
        figz = (6.2, 3.25)
        fig = plt.figure(1, figsize=figz)
        fz = 18
        lz = 16

        if variable == 'Pa':
            data = pd.read_csv(TABDIR + 'SynchInt1MPade(4;0,2,3)INFPFirstAandB030226.csv')
            plotcolor = 'C0'
            def model(x):
                return a_parameter(x)
        elif variable == 'Pb':
            data = pd.read_csv(TABDIR + 'SynchInt1MPade(4;0,2,3)INFPFirstAandB030226.csv')
            plotcolor = 'C1'
            def model(x):
                return b_parameter(x)
        elif variable == 'Pn':
            data = pd.read_csv(TABDIR + 'SynchInt1MPade(4;0,2,3)INFPFirst030226.csv')
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
                #axs[i,j].set_yscale('log')
                axs[i,j].set_xlabel(r"$\chi_e$", fontsize=fz)


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
            axs[0,0].set_title(r'$a(\chi_e)$', fontsize=(fz))
        if variable == 'Pb':
            axs[0,0].set_title(r'$b(\chi_e)$', fontsize=(fz))
        if variable == 'Pn':
            axs[0,0].set_title(r'$n(\chi_e)$', fontsize=(fz))
            #axs[0,0].set_yscale('log')



        axleft    = 0.14
        axbottom  = 0.2
        axright   = 0.94
        axtop     = 0.9

        fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)

        name = 'QuarticSynch' + variable + 'param160526julkaisu.pdf'#+ 'param100226.pdf'
        plt.savefig(FIGDIR + name)
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

        rvalues = np.linspace(0.01, 0.99, 50)


        axs[0,0].plot(chi, abs(error/Numeric_values),
                    color='magenta',
                    alpha = 1.0,
                    lw = 1.0,
                    linestyle='solid'
                    )


        for j in range(ncol_fig):
            for i in range(nrow_fig):
                axs[i,j].set_xscale('log')
                axs[i,j].set_xlabel(r"$\chi_e$", fontsize=fz)
                axs[i,j].set_yscale('log')

        axs[0,0].set_title(r'$\mathrm{Relative~ error}$', fontsize=16)

        name = 'QuarticSynch' + variable + 'paramrelerror160526julkaisu.pdf'
        


        axleft    = 0.14
        axbottom  = 0.2
        axright   = 0.94
        axtop     = 0.9

        fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)
        plt.savefig(FIGDIR + name)
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

        rvalues = np.linspace(0.01, 0.99, 50)


        axs[0,0].plot(chi, abs(error),
                    color='limegreen',
                    alpha = 1.0,
                    lw = 1.0,
                    linestyle='solid'
                    )


        for j in range(ncol_fig):
            for i in range(nrow_fig):
                axs[i,j].minorticks_on()
                axs[i,j].set_xlabel(r"$\chi_e$", fontsize=fz)
                axs[i,j].set_xscale('log')
                axs[i,j].set_yscale('log')

        axs[0,0].set_title(r'$\mathrm{Absolute~ error}$', fontsize=16)


        axleft    = 0.14
        axbottom  = 0.2
        axright   = 0.94
        axtop     = 0.9

        fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)

        name = 'QuarticSynch' + variable + 'paramabserror160526julkaisu.pdf'
        plt.savefig(FIGDIR + name)

        plt.show()


# The values of the auxiliary parameters gathered from the probability (Pa, Pb, Pn) and the radiation power spectrum (Na, Nb, Nn)

if False:

    data = pd.read_csv(TABDIR + 'SynchInt1MPade(4;0,2,3)INFPFirst030226.csv')
    fig = plt.figure(1, figsize=(7.5, 2.5))

    fz = 17

    # add ticks to both sides 
    plt.rc('xtick', top = True)
    plt.rc('ytick', right = True)

    plt.rc('font',  family='serif')
    plt.rc('text',  usetex=True)

    # make labels slightly smaller
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes',  labelsize=15)
    plt.rc('legend',  handlelength=2.0)

    # number of rows and columns for the figure
    nrow_fig = 1
    ncol_fig = 3

    gs = plt.GridSpec(nrow_fig, ncol_fig)
    gs.update(wspace = 0.25)
    gs.update(hspace = 0.4)

    axs = np.empty( (nrow_fig,ncol_fig), dtype=object)
    ticklevels = [1e-3, 1e-1, 1e1]

    for j in range(ncol_fig):
        for i in range(nrow_fig):
            axs[i,j] = plt.subplot(gs[i,j])
            axs[i,j].minorticks_on()
            axs[i,j].set_xscale('log')
            axs[i,j].set_xlabel(r"$\chi_e$", fontsize=fz)
            #axs[i,j].set_xticklabels(['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])
            plt.xticks(np.array(ticklevels))
            #axs[i,j].set_yscale('log')

    rvalues = np.linspace(0.01, 0.99, 50)

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


    axs[0,0].set_title(r'$a(\chi_e)$', fontsize=fz)
    #axs[0,0].set_yscale('log')
    #axs[0,1].set_yscale('log')
    axs[0,1].set_title(r'$b(\chi_e)$', fontsize=fz)
    axs[0,2].set_title(r'$n(\chi_e)$', fontsize=fz)


    axleft    = 0.14
    axbottom  = 0.3
    axright   = 0.94
    axtop     = 0.9

    fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)


    fig.savefig(FIGDIR + 'SynchABNparamplot100226julkaisu.pdf')
    plt.show()



# Using the derived expression for the inverted cumulative probability 
# to plot the distribution of r-values of the emitted photons


if False:
    epsilon = 0.0001

    fig = plt.figure(1, figsize=(6.7, 5.5))
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

    for j in range(ncol_fig):
        for i in range(nrow_fig):
            axs[i,j] = plt.subplot(gs[i,j])
            axs[i,j].minorticks_on()
            axs[i,j].set_xlabel(r'$\chi_\gamma/\chi_e$', fontsize=14)
            axs[i,j].set_yscale('log')
            #axs[i,j].set_xscale('log')

    for n, chi_in in enumerate([0.0005, 0.05, 0.5, 50]):


        N = 300**2
        Np = int(np.sqrt(N))
        chi_outs = np.zeros(N)

        for i in range(N):
            zeta = np.random.uniform(low=Z_CLIP, high=(1 - Z_CLIP))
            r = r_solutions(zeta, chi_in)[1]
            chi_outs[i] = r

        H, bins_out = np.histogram(chi_outs, bins=int(np.sqrt(N)), density=True)
        bins_center = (bins_out[1:] + bins_out[:-1]) / 2

        min_r = min(chi_outs)
        max_r = max(chi_outs)
        Nr = 100000


        if n == 0: 
            r_values = np.linspace(epsilon*0.01, 1-epsilon, 5*Nr)#epsilon*0.01, 0.1-epsilon1000#Np + 1)#np.linspace(min_r, max_r, (Np + 1))
            row = 0
            column = 0
            y0 = 5e-3
            y1 = 1e4
            x1 = 0.008
        if n == 1:
            r_values = np.linspace(epsilon*0.01, 1-epsilon*0.01, Nr) # Np+1
            row = 0
            column = 1
            y0 = 3e-3
            y1 = 5e2
            x1 = 0.35
        if n == 2:
            r_values = np.linspace(epsilon, 1-epsilon, Nr)
            row = 1
            column = 0
            y0 = 3e-3
            y1 = 2e2
            x1 = 0.88
        if n == 3:
            r_values = np.linspace(epsilon, 1-epsilon, Nr)
            row = 1
            column = 1
            y0 = 2e-2
            y1 = 9e1
            x1 = 1
        


        T = T_Synchrotron(chi_in)

        def P_SyncExct(x, r):
            return N_Synchrotron(x, x * r) / T#/ T_Synchrotron(x)

        #p_values = np.diff(np.array([P_SyncExct(chi_in, r) for r in r_values]))
        dr = r_values[1] - r_values[0]
        r_avg = (r_values[1:] + r_values[:-1])/2
        d_avg = r_avg[1]- r_avg[0]
        N_spectrum = np.array([N_Synch_spectrum(chi_in, chi_in*r) for r in r_values])
        N_numeric = N_spectrum / (np.sum(N_spectrum)*dr)
        N_numeric2 = N_spectrum / (np.sum(N_spectrum)*d_avg)


        min_pval = min(r_values)
        smaller_r0 = chi_outs[chi_outs < min_pval]
        smaller_r1 = chi_outs[chi_outs < min_pval*0.1]
        smaller_r2 = chi_outs[chi_outs < min_pval*0.01]

        n_chi_outs = len(chi_outs)

        print(r'chi = ', (chi_in))
        print('Frac of smaller than min from p_values: ', len(smaller_r0) / n_chi_outs)
        print('Frac of smaller than min from 0.1*p_values: ', len(smaller_r1) / n_chi_outs)
        print('Frac of smaller than min from 0.001*p_values: ', len(smaller_r2) / n_chi_outs)

        #axs[row, column].plot(r_avg, p_values/np.sum(p_values*d_avg), color='red', linestyle='--')
        axs[row, column].set_title(f'$\\chi_e = {chi_in}$', fontsize=14)
        axs[row, column].plot(r_values, N_numeric, color='black', linestyle='solid')
        #axs[row, column].plot(r_values, N_numeric2, color='C2', linestyle=':')
        #axs[row, column].plot(bins_center, H, color='Orange', drawstyle='steps') # In BW case this seemed to give solutions skewed to the left???
        axs[row, column].hist(chi_outs, bins = int(np.sqrt(N)), histtype='step', color='C1', density=True) # Seems to give the correct centering originally bins = int(np.sqrt(N))
        axs[row, column].set_xlim(0, x1)#(min_r, max_r) # Could just collect the limits from the figure by hand?
        axs[row, column].set_ylim(y0, y1)
    axleft    = 0.14
    axbottom  = 0.2
    axright   = 0.94
    axtop     = 0.9

    fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)
    #plt.savefig(FIGDIR + 'SynchProducedSpectrajulkaisu.pdf')
    plt.show()


