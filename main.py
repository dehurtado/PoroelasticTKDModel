#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python codes for the TKD poroelastic model for lung parenchyma, as described in paper "Upscaling the poroelastic behavior of the lung parenchyma: A finite-deformation micromechanical model" by Felipe Concha and Daniel E. Hurtado.

Find more in
https://www.sciencedirect.com/science/article/abs/pii/S0022509620303811
"""

import numpy as np
import matplotlib.pyplot as plt
import TKDlib as tkdl

# material parameters 
fo = 0.69
mu = 31.    # KPa
material_params  = {'material_model_name': 'Incompressible Neo-Hookean',
                    'mu':mu,
                    'fo':fo}
# model parameters
d  = 0.55
alpha = 6
model_params = {'d':d, 'alpha':alpha}

# plot parameters
nsteps = 10
text_size = 20
lw = 2.
legend_size = 13


def TKDStressStretch(lam1c, lam2c, lam3c, pressure, model_params, material_params, nsteps):
    
    stretch = []; sig11 = []; sig22 = []; sig33 = []
    tkd    = tkdl.TKD(model_params, material_params)

    # get stress-stretch data
    for i in np.arange(nsteps+1):
        lam1_i = 1. + (lam1c - 1.)*i/nsteps
        lam2_i = 1. + (lam2c - 1.)*i/nsteps
        lam3_i = 1. + (lam3c - 1.)*i/nsteps
        F = np.diag([lam1_i, lam2_i, lam3_i])
        
        q, SIG = tkd.solveTKD(F, pressure)
        sig11.append(SIG[0,0])
        sig22.append(SIG[1,1])
        sig33.append(SIG[2,2])
        stretch.append(lam1_i)

    stretch = np.asarray(stretch)        
    sig11 = np.asarray(sig11)
    sig22 = np.asarray(sig22)
    sig33 = np.asarray(sig33)
        
    return stretch, sig11, sig22, sig33
    


# =============================================================================
# Isotropic stretching example
# =============================================================================

# input
lam1c = 1.3
lam2c = 1.3
lam3c = 1.3
pressure = 0.4

# get data
stretch, sig11, sig22, sig33 = TKDStressStretch(lam1c, lam2c, lam3c, pressure, model_params, material_params, nsteps)

# plot curve
sigma_hydro = 1./3.*(sig11 + sig22 + sig33)
plt.figure(1)
plt.plot(stretch, 1./mu*sigma_hydro, color='black', label='TKD: '+r"$tr \ {\bf \sigma^c} /3\mu$",linestyle='-',linewidth=lw)

plt.rcParams['figure.dpi'] = 300
plt.xlabel(r"$\lambda_1^c$",size=text_size)
plt.ylabel('Cauchy stress'+r"$/\mu$",size=text_size)
plt.grid(which='both',linewidth=0.5,linestyle='--')
plt.legend(loc='upper left', prop={'size': legend_size})

plt.xlim([1., 1.3])
plt.ylim([-0.02, 0.08 ])
plt.xticks(np.arange(1.,1.3,0.1))
plt.yticks(np.arange(-0.02,0.082,0.02))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# =============================================================================
# Anisotropic stretching example
# =============================================================================

# input
lam1c = 1.7
lam2c = 1.3
lam3c = 1.
pressure = 0.4

# get data
plt.figure(2)
stretch, sig11, sig22, sig33 = TKDStressStretch(lam1c, lam2c, lam3c, pressure, model_params, material_params, nsteps)

# plot curve
plt.plot(stretch, 1./mu*sig11, color='black', label='TKD: '+r"$\sigma_{11}^c/\mu$", linestyle='-',linewidth=lw)  
plt.plot(stretch, 1./mu*sig22, color='red'  , label='TKD: '+r"$\sigma_{22}^c/\mu$", linestyle='-',linewidth=lw)  
plt.plot(stretch, 1./mu*sig33, color='blue' , label='TKD: '+r"$\sigma_{33}^c/\mu$", linestyle='-',linewidth=lw)  

plt.rcParams['figure.dpi'] = 300
plt.xlabel(r"$\lambda_1^c$",size=text_size)
plt.ylabel('Cauchy stress'+r"$/\mu$",size=text_size)
plt.grid(which='both',linewidth=0.5,linestyle='--')
plt.legend(loc='upper left', prop={'size': legend_size})

plt.xlim([1., 1.3])
plt.ylim([-0.02, 0.08 ])
plt.xticks(np.arange(1.,1.3,0.1))
plt.yticks(np.arange(-0.02,0.082,0.02))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# =============================================================================
# Biaxial stretching example
# =============================================================================

# input
lam1c = 1.5
lam2c = 1.5
lam3c = 1.
pressure = 0.4

# get data
plt.figure(3)
stretch, sig11, sig22, sig33 = TKDStressStretch(lam1c, lam2c, lam3c, pressure, model_params, material_params, nsteps)

# plot curve
sig11_22_avg = 1./2.*(sig11 + sig22)
plt.plot(stretch, 1./mu*sig11_22_avg, color='black', label='TKD: '+r"$(\sigma_{11}^c+\sigma_{22}^c)/2\mu$", linestyle='-',linewidth=lw)  
plt.plot(stretch, 1./mu*sig33, color='red' , label='TKD: '+r"$\sigma_{33}^c/\mu$", linestyle='-',linewidth=lw)  

plt.rcParams['figure.dpi'] = 300
plt.xlabel(r"$\lambda_1^c$",size=text_size)
plt.ylabel('Cauchy stress'+r"$/\mu$",size=text_size)
plt.grid(which='both',linewidth=0.5,linestyle='--')
plt.legend(loc='upper left', prop={'size': legend_size})

plt.xlim([1., 1.3])
plt.ylim([-0.02, 0.08 ])
plt.xticks(np.arange(1.,1.3,0.1))
plt.yticks(np.arange(-0.02,0.082,0.02))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# =============================================================================
# Uniaxial stretching example
# =============================================================================

# input
lam1c = 1.8
lam2c = 1.
lam3c = 1.
pressure = 0.4

# get data
plt.figure(4)
stretch, sig11, sig22, sig33 = TKDStressStretch(lam1c, lam2c, lam3c, pressure, model_params, material_params, nsteps)

# plot curve
sig22_33_avg = 1./2.*(sig11 + sig22)
plt.plot(stretch, 1./mu*sig11, color='black', label='TKD: '+r"$\sigma_{11}^c/\mu$", linestyle='-',linewidth=lw)  
plt.plot(stretch, 1./mu*sig22_33_avg, color='red', label='TKD: '+r"$(\sigma_{22}^c+\sigma_{33}^c)/2\mu$", linestyle='-',linewidth=lw)  

plt.rcParams['figure.dpi'] = 300
plt.xlabel(r"$\lambda_1^c$",size=text_size)
plt.ylabel('Cauchy stress'+r"$/\mu$",size=text_size)
plt.grid(which='both',linewidth=0.5,linestyle='--')
plt.legend(loc='upper left', prop={'size': legend_size})

plt.xlim([1., 1.3])
plt.ylim([-0.02, 0.08 ])
plt.xticks(np.arange(1.,1.3,0.1))
plt.yticks(np.arange(-0.02,0.082,0.02))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
    