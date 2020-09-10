#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:55:08 2020

@author: felipeconcha
"""

# =============================================================================
# Packages
# =============================================================================
import numpy as np
import sympy as sp
import TKDlib as tkdl

# =============================================================================
# Parameters
# =============================================================================
# material parameters
mu = sp.Symbol('mu')
fo = sp.Symbol('fo')
material_params  = {'material_model_name': 'Incompressible Neo-Hookean',
                    'mu':mu, 'fo':fo}

# model parameters
d  = sp.Symbol('d')
alpha = sp.Symbol('alpha')
# kr = sp.Symbol('kr')
tkd_model_params = {'d':d, 'alpha':alpha}

# =============================================================================
# INPUTS
# =============================================================================

# degrees of freedom
q1 = sp.Symbol('q1y')
q2 = sp.Symbol('q2x')
q3 = sp.Symbol('q3z')
r = np.array([q1, q2, q3])

# imposed stretches
lam1 = sp.Symbol('lam1')
lam2 = sp.Symbol('lam2')
lam3 = sp.Symbol('lam3')
LAM = np.array([lam1, lam2, lam3])
F = np.diag([lam1, lam2, lam3])

# imposed pressure
pressure = sp.Symbol('pressure')

# =============================================================================
# Setting the model up
# =============================================================================

# compute symbolic TKD values
tkd = tkdl.TKD(tkd_model_params, material_params, symbolic=True)
tkd.setDeformedGeometry(r, F)
tkd.setAxialEnergy()
tkd.setRotationalEnergy()
tkd.setExternalEnergy(pressure)
PI = tkd.Eaxial_tot + tkd.Erot_tot - tkd.Eext_tot

# forces
L   = tkd.strutsLengths(xyz=tkd.xyz)    
lam = tkd.strutsStretches(L=L)
A   = tkd.strutsAreas(lam=lam)
direc = tkd.strutsDirections(xyz=tkd.xyz)
fvec = tkd.strutsForces(A=A, lam=lam, direc=direc, pressure=pressure)
f7  = fvec[6,:]    
f10 = fvec[9,:]    
f15 = fvec[14,:]

# directions
# # Useful for averaging stresses
# direc = tkd.strutsDirections(tkd.xyz)
# a1_direc, a2_direc = tkd.strutsOrthogonalDirections(direc)

        
# =============================================================================
# Compute quantities of interest
# =============================================================================

# residual
print('Computing dVdq ... ')
R = sp.diff(PI, r)

# tangent stiffness matrix
print('\n')
print('Computing jacobian ...')
neq = np.shape(R)[0]
K = sp.zeros(neq, neq)
for i in np.arange(neq):
    for j in np.arange(neq):
        print('Computing jacobian, component: ('+str(i)+','+str(j)+')')
        K[i, j] = sp.diff(R[i], r[j])
 
        
# =============================================================================
# # write equations in txt
# =============================================================================

print('Writing TKD equations ...')

solution_eqn = [R, K]
solution_sn = ['R','J']

# write .py function
var = 'mu, fo, d, alpha, q1y, q2x, q3z, lam1, lam2, lam3, pressure'
# var = 'mu, fo, d, kr, q1y, q2x, q3z, lam1, lam2, lam3'
tkdl.Utilities.writeEquations_py([R], ['R'], var, 'R', './TKD_residual3.py') #eqn, name, variables, output, path
tkdl.Utilities.writeEquations_py([K], ['J'], var, 'J', './TKD_jacobian3.py') #eqn, name, variables, output, path

# write text files
file_name_solution = './' + material_params['material_model_name'] + '_solution.txt'
tkdl.Utilities.writeEquations_txt(solution_eqn, solution_sn, file_name_solution)

# write .py function
f = solution_eqn + forces_eqn + d2PIdrdlam_eqn
sn = solution_sn + forces_sn + d2PIdrdlam_sn
varout = 'R, J'
path = './'
tkdl.Utilities.writeEquations_py(solution_eqn,solution_sn,var,varout,path)







