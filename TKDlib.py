#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:21:32 2019

@author: cb2l-001
"""

#from __future__ import division  # useful for symbolic division without dots (3/2 = 1.5)
import numpy as np
import sympy as sp
from numpy import sqrt as sqrt
from numpy import exp as exp
from scipy.optimize import fsolve
import matplotlib.pylab as plt

import TKD_residual as tkd_residual
import TKD_jacobian as tkd_jacobian


class TKD:

    def __init__(self, model_params, material_params, symbolic=False):

        """
        TKD is solved for the TKD region, composed by 18 nodes
        """
        
        # parameters
        self.mu = material_params['mu']
        self.fo = material_params['fo']
        self.material_model_name = material_params['material_model_name']
        self.d  = model_params['d']
        self.alpha = model_params['alpha']
        self.symbolic = symbolic

        # initial volume -----------------------------------------------------
        a     = 1.
        Eo    = a / 3. * np.sqrt(2.)              # strut length
        delta = Eo / 2.                           # half strut length
        self.Vo = 64. * np.sqrt(2.) * delta ** 3  # initial TKD volume
        
        
        # initial strut area -------------------------------------------------
        Vso     = (1. - self.fo) * self.Vo
        self.Ao = Vso / ((12. - 9. * self.d) * Eo)
        
        
        # rotational stiffness -----------------------------------------------
        I = self.Ao**2 / (4.*np.pi)
        E = 3. * self.mu
        self.kr = self.alpha * (E * I) / (Eo)
        
        
        # lattice vectors ----------------------------------------------------
        self.bx = 2. * np.sqrt(2.) * delta * np.array([-1., 1., 1.])
        self.by = 2. * np.sqrt(2.) * delta * np.array([1., -1., 1.])
        self.bz = 2. * np.sqrt(2.) * delta * np.array([1., 1., -1.])
        
        
        # TKDr coordinates ---------------------------------------------------
        x1 = np.array([0., 1. / 3., 2. / 3.])
        x2 = np.array([1. / 3., 0., 2. / 3.])
        x3 = np.array([2. / 3., 0., 1. / 3.])
        x4 = np.array([2. / 3., 1. / 3., 0.])
        x5 = np.array([1. / 3., 2. / 3., 0.])
        x6 = np.array([0., 2. / 3., 1. / 3.])
        x7 = np.array([0., 1. / 2., 5. / 6.])
        x8 = np.array([-1. / 6., 1. / 6., 2. / 3])
        x9 = np.array([1. / 6., -1. / 6., 2. / 3.])
        x10 = np.array([1. / 2., 0., 5. / 6.])
        x11 = np.array([2. / 3., -1. / 6., 1. / 6.])
        x12 = np.array([5. / 6., 0., 1. / 2.])
        x13 = np.array([2. / 3., 1. / 6., -1. / 6.])
        x14 = np.array([5. / 6., 1. / 2., 0.])
        x15 = np.array([1. / 2., 5. / 6., 0.])
        x16 = np.array([1. / 6., 2. / 3., -1. / 6.])
        x17 = np.array([0., 5. / 6., 1. / 2.])
        x18 = np.array([-1. / 6., 2. / 3., 1. / 6.])

        self.xyzo = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, 
                              x12, x13, x14, x15, x16, x17, x18])
        self.ien = np.array([[2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [1, 6], 
                             [7, 1], [8, 1], [9, 2], [10, 2], [11, 3], 
                             [12, 3], [13, 4], [14, 4], [15, 5], [16, 5], 
                             [17, 6], [18, 6]]) - 1
        
        
        # initial lengths ----------------------------------------------------
        self.Lo = self.strutsLengths(self.xyzo)
        
        
        # rotations ----------------------------------------------------------
        self.angstrut = np.array([[6,7,8],\
                             [1,9,10],\
                             [2,11,12],\
                             [3,13,14],\
                             [4,15,16],\
                             [5,17,18],\
                             [1,6,8],\
                             [1,6,7],\
                             [1,2,10],\
                             [1,2,9],\
                             [2,3,12],\
                             [2,3,11],\
                             [3,4,14],\
                             [3,4,13],\
                             [4,5,16],\
                             [4,5,15],\
                             [5,6,18],\
                             [5,6,17]])-1

        # where are the rotational springs?
        # next matrix indicates the row and the column of each rotation of interest
        self.rotspring = np.array([[1,1],[1,2],[1,3],[7,2],[7,3],[8,2],\
                              [2,1],[2,2],[2,3],[9,1],[9,3],[10,1],\
                              [3,1],[3,2],[3,3],[11,1],[11,3],[12,1],\
                              [4,1],[4,2],[4,3],[13,1],[13,3],[14,1],\
                              [5,1],[5,2],[5,3],[15,1],[15,3],[16,1],\
                              [6,1],[6,2],[6,3],[17,1],[17,3],[18,1]])-1
            
        # initial angles
        self.ango = self.strutsRotation(self.xyzo)
            

    def nodalDisplacements(self, q, F):

        """
        Function that computes all the nodal displacements of TKD region (TKDr)
        from the independent degrees of freedom q1y,q2x,q3z.
        """
        q1y, q2x, q3z = q
        lam1, lam2, lam3 = F.diagonal()

        # node 1
        q1x = 0.
        q1y = q1y
        q1z = (1. - lam3) * self.bz[2]
        q1 = np.array([q1x, q1y, q1z])

        # node 2
        q2x = q2x
        q2y = 0.
        q2z = (1. - lam3) * self.bz[2]
        q2 = np.array([q2x, q2y, q2z])

        # node 3
        q3x = (1. - lam1) * self.bx[0]
        q3y = 0.
        q3z = q3z
        q3 = np.array([q3x, q3y, q3z])

        # node 4
        q4x = (1. - lam1) * self.bx[0]
        q4y = -q1y + (lam2 - 1.) * self.bz[1]
        q4z = 0.
        q4 = np.array([q4x, q4y, q4z])

        # node 5
        q5x = -q2x + (1. - lam1) * self.bx[0]
        q5y = (1. - lam2) * self.by[1]
        q5z = 0.
        q5 = np.array([q5x, q5y, q5z])

        # node 6
        q6x = 0.
        q6y = (1. - lam2) * self.by[1]
        q6z = (1. - lam3) * self.by[1] - q3z
        q6 = np.array([q6x, q6y, q6z])

        # node 11
        q11x = (1. - lam1) * self.bx[0]
        q11y = (q1y - (1. - lam2) * self.bx[0]) / 2.
        q11z = q3z / 2.
        q11 = np.array([q11x, q11y, q11z])

        # node 7
        q7x = 0.
        q7y = (q1y + (1. - lam2) * self.bx[0]) / 2.
        q7z = q3z / 2. + (1. - lam3) * self.bx[0]
        q7 = np.array([q7x, q7y, q7z])

        # node 8
        q8x = -q2x / 2.
        q8y = q1y / 2.
        q8z = (1. - lam3) * self.bz[2]
        q8 = np.array([q8x, q8y, q8z])

        # node 9
        q9x = q2x / 2.
        q9y = -q1y / 2.
        q9z = (1. - lam3) * self.bz[2]
        q9 = np.array([q9x, q9y, q9z])

        # node 10
        q10x = (q2x + (1. - lam1) * self.bx[0]) / 2.
        q10y = 0.
        q10z = 3. / 2. * (1 - lam3) * self.bz[2] - q3z / 2.
        q10 = np.array([q10x, q10y, q10z])

        # node 12
        q12x = 3. / 2. * (1. - lam1) * self.bx[0] - q2x / 2.
        q12y = 0.
        q12z = q3z / 2. + (1. - lam3) * self.bz[2] / 2.
        q12 = np.array([q12x, q12y, q12z])

        # node 13
        q13x = (1. - lam1) * self.bx[0]
        q13y = 1. / 2. * (-q1y + (1. - lam2) * self.bz[2])
        q13z = -q3z / 2.
        q13 = np.array([q13x, q13y, q13z])

        # node 14
        q14x = q2x / 2. + (1. - lam1) * self.bz[2]
        q14y = -q1y / 2. + (1. - lam2) * self.bz[2]
        q14z = 0.
        q14 = np.array([q14x, q14y, q14z])

        # node 15
        q15x = -q2x / 2. + (1. - lam1) * self.bz[2]
        q15y = q1y / 2. + (1. - lam2) * self.bz[2]
        q15z = 0.
        q15 = np.array([q15x, q15y, q15z])

        # node 16
        q16x = ((1. - lam1) * self.by[1] - q2x) / 2.
        q16y = (1. - lam2) * self.by[1]
        q16z = (q3z - (1. - lam3) * self.bz[2]) / 2.
        q16 = np.array([q16x, q16y, q16z])

        # node 17
        q17x = 0.
        q17y = (-q1y + 3. * (1. - lam2) * self.by[1]) / 2.
        q17z = -q3z / 2. + (1. - lam3) * self.bx[0]
        q17 = np.array([q17x, q17y, q17z])

        # node 18
        q18x = (q2x - (1. - lam1) * self.bx[0]) / 2.
        q18y = (1. - lam2) * self.by[1]
        q18z = (-q3z + (1. - lam3) * self.bz[2]) / 2.
        q18 = np.array([q18x, q18y, q18z])

        qvertices = np.array([q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18])
        

        return qvertices


    def setDeformedGeometry(self, q, F):

        """
        Function that computes the deformed geometry of TKDr from the imposed
        deformation gradient and the solved degrees of freedom

        xyz: nodal coordinates of TKDr for deformed configuration (18 nodes)
        """

        # nodal displacements as function of F
        qvertices = self.nodalDisplacements(q, F)

        # current coordinates
        self.xyz = self.xyzo + qvertices
        
    def strutsLengths(self, xyz):
        """
        Function that computes the length of each strut of TKD

        return:
        L: matrix or vector (depending on symbolic or not) of strut lengths
        """
        nelem = np.shape(self.ien)[0]

        if not self.symbolic:
            L = np.zeros(nelem)
            for i in np.arange(nelem):
                xi = xyz[self.ien[i,1],:]
                xj = xyz[self.ien[i,0],:]
                L[i] = np.sqrt( (xj[0]-xi[0])**2 + (xj[1]-xi[1])**2 + (xj[2]-xi[2])**2 )

        else:
            print('Computing symbolic struts length')
            L = sp.zeros(nelem,1)
            for i in np.arange(nelem):
                xi = xyz[self.ien[i,1],:]
                xj = xyz[self.ien[i,0],:]
                L[i,0] = sp.sqrt( (xj[0]-xi[0])**2 + (xj[1]-xi[1])**2 + (xj[2]-xi[2])**2 )

        return L


    def strutsStretches(self, L):

        """ It compute stretches in every strut of TKD:
        Lo: vector of initial strut lengths
        L: vector of current strut length

        return:
        v: elongation
        lam: deformation
        """
        # current lengths
        if not self.symbolic:
            # v = L-Lo
            lam = L/self.Lo

        else:
            print('Computing symbolic struts stretches')
            nelem = self.Lo.shape[0]
            # v   = sp.zeros(nelem,1)
            lam = sp.zeros(nelem,1)
            for i in np.arange(nelem):
                # v[i,0]   = L[i,0] - Lo[i]
                lam[i,0] = L[i,0]/self.Lo[i]

        return lam

    def strutsAreas(self, lam):

        """ It computes current areas in every strut of TKD
        """

        if not self.symbolic:
            A = self.Ao/lam

        else:
            print('Computing symbolic struts area')
            nelem = lam.shape[0]
            A = sp.zeros(nelem,1)
            for i in np.arange(nelem):
                A[i] = self.Ao/lam[i,0]

        return A

    def constitutiveRelationForStrut(self, lami, pressure):

        """
        :param lami: stretch for i-th strut
        """

        if not self.symbolic:

            if self.material_model_name == 'Incompressible Neo-Hookean':
                p = self.mu / lami + pressure * (2. - np.sqrt(lami))
                sig_axial = self.mu * lami**2. - p   
                energy = self.mu*(lami**2/2. + 1./lami - 3./2.)

            else:
                raise Exception('UE: material model not recognized')

        else:

            if self.material_model_name == 'Incompressible Neo-Hookean':
                p = self.mu / lami + pressure * (2. - sp.sqrt(lami))
                sig_axial = self.mu * lami ** 2. - p  
                energy = self.mu/2.*(lami**2. + 2./lami - 3.)

            else:
                raise Exception('User error: material model not recognized')

        return sig_axial, energy

    def strutsEnergy(self, lam):

        """
        Energy for the Neo-Hookean material model
        """
        Et = 0.

        if not self.symbolic:
            nelem = np.shape(lam)[0]
            E = np.zeros(nelem)
            for i in np.arange(nelem):
                _, energy = self.constitutiveRelationForStrut(lam[i], 0.)
                E[i] = self.Ao * self.Lo[i] * energy
            Et = np.sum(E)

        else:
            print('Computing symbolic struts axial energy')
            nelem = lam.shape[0]
            E  = sp.zeros(nelem)
            for i in np.arange(nelem):
                _, energy = self.constitutiveRelationForStrut(lam[i,0], 0.)
                E[i,0] = self.Ao * self.Lo[i] * energy
                Et = Et + E[i,0]

        return E, Et
    
    def strutsDirections(self, xyz):

        """ It computes the current direction of struts in TKD
        """
        nelem = np.shape(self.ien)[0]

        if not self.symbolic:
            direc = np.zeros([nelem,3])
            for i in np.arange(nelem):
                xi = xyz[self.ien[i,1],:]
                xj = xyz[self.ien[i,0],:]
                D = xj-xi
                Dn = np.linalg.norm(D)
                direc[i,:] = D/Dn

        else:
            print('Computing symbolic struts direction')
            direc = sp.zeros(nelem,3)
            for i in np.arange(nelem):
                xi = xyz[self.ien[i,1],:]
                xj = xyz[self.ien[i,0],:]
                D = xj-xi
                Dn = sp.sqrt(D[0]**2 + D[1]**2 + D[2]**2)
                for j in np.arange(3):
                    direc[i,j] = D[j]/Dn

        return direc
    
    def strutsOrthogonalDirections(self, direc):

        nelem = np.shape(direc)[0]
        ez = np.array([0.,0.,1.])
        a1_direc = np.zeros([nelem, 3])
        a2_direc = np.zeros([nelem, 3])
        for i in np.arange(nelem):
            t = direc[i,:]
            a1_direc[i,:] = np.cross(ez, t)
            a2_direc[i,:] = np.cross(t, a1_direc[i,:])

        return a1_direc, a2_direc
    
    def strutsEffectiveLengthFactor(self):

        
        # current lengths
        nelem = 18
        if not self.symbolic:
            eff_length_fact = np.ones(nelem)
            for i in np.arange(6, nelem):
                eff_length_fact[i] = 1. - self.d

        else:
            print('Computing symbolic struts stretches')
            eff_length_fact = sp.ones(nelem)
            for i in np.arange(6, nelem):
                eff_length_fact[i] = 1. - self.d

        return eff_length_fact
    
    def strutsForces(self, A, lam, direc, pressure):

        """ It compute axial forces in each strut of TKD
        """

        if not self.symbolic:
            nelem = np.shape(lam)[0]
            fvec = np.zeros([nelem,3])
            for i in np.arange(nelem):
                fi, _ = self.constitutiveRelationForStrut(lam[i], pressure)
                fvec[i,:] = A[i] * fi * direc[i,:]

        else:
            print('Computing symbolic struts forces')

            nelem = lam.shape[0]
            fvec = sp.zeros(nelem,3)
            for i in np.arange(nelem):
                fi, _ = self.constitutiveRelationForStrut(lam[i,0], pressure)
                for j in np.arange(3):
                    fvec[i,j] = A[i,0] * fi * direc[i,j]
  
        return fvec

    def strutsRotation(self, xyz):
        
        # compute angles
        if not self.symbolic:
            ang = np.zeros([18,3])
            for i in np.arange(18):
                a = xyz[self.ien[i,0],:] - xyz[self.ien[i,1],:]
                a = a/np.linalg.norm(a)
                
                for j in np.arange(3):
                    el = self.angstrut[i,j]
                    b = xyz[self.ien[el,0],:] - xyz[self.ien[el,1],:]
                    b = b/np.linalg.norm(b)
                    
                    ang[i,j] = np.dot(a,b)
                    #ang[i,j] = np.arccos(np.dot(a,b))
                    
        else:
            ang = sp.zeros(18,3)
            for i in np.arange(18):
                a = xyz[self.ien[i,0],:] - xyz[self.ien[i,1],:]
                moda = sp.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
                a = a/moda
                
                for j in np.arange(3):
                    el = self.angstrut[i,j]
                    b = xyz[self.ien[el,0],:] - xyz[self.ien[el,1],:]
                    modb = sp.sqrt(b[0]**2 + b[1]**2 + b[2]**2)
                    b = b/modb
                    
                    ang[i,j] = sp.simplify(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])
                    # ang[i,j] = sp.acos(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])
    
        return ang
    
    def setRotationalEnergy(self):
    
        Vr = 0.
        ang = self.strutsRotation(self.xyz)
        if not self.symbolic:
            Vri = np.zeros(36)
            for i in np.arange(36):
                Vri[i] = 1./2 * self.kr * (ang[self.rotspring[i,0], self.rotspring[i,1]] - 
                                           self.ango[self.rotspring[i,0], self.rotspring[i,1]])**2
                Vr = Vr + Vri[i]
                
        else:
            Vri = sp.zeros(36,1)
            for i in np.arange(36):
                Vri[i,0] = sp.simplify(1./2 * self.kr * (ang[self.rotspring[i,0], self.rotspring[i,1]]-
                                           self.ango[self.rotspring[i,0], self.rotspring[i,1]])**2)
                Vr = Vr + Vri[i,0]
       
        self.Erot = Vri
        self.Erot_tot = Vr

    def setExternalEnergy(self, pressure):

        Pi_ext_t = 0.
        nelem = np.shape(self.lam)[0]

        if not self.symbolic:
            Pi_ext = np.zeros(nelem)
            for i in np.arange(nelem):
                Pi_ext[i] = -2. * pressure * self.Ao * self.Lo[i] * (1. - np.sqrt(self.lam[i]))
            Pi_ext_t = np.sum(Pi_ext)

        else:
            print('Computing symbolic struts direction')
            Pi_ext = sp.zeros(nelem, 3)
            for i in np.arange(nelem):
                Pi_ext[i, 0] = -2. * pressure * self.Ao * self.Lo[i] * (1. - sp.sqrt(self.lam[i, 0]))
                Pi_ext_t = Pi_ext_t + Pi_ext[i, 0]

        self.Eext = Pi_ext
        self.Eext_tot = Pi_ext_t
        
    def setAxialEnergy(self):

        """
        After calculating the current geometry, we should be able to obtain 
        the axial elastic potential enegy per strut and total
        Set:
        - Current stretches
        - Current cross sectional areas
        - Current axial energy (per struts and total)
        """

        #struts length
        self.L   = self.strutsLengths(self.xyz)
        self.lam = self.strutsStretches(self.L)
        self.A   = self.strutsAreas(self.lam)
        self.Eaxial, self.Eaxial_tot = self.strutsEnergy(self.lam)

    def evalResidual(self, q, F, pressure):
                
        # stretches
        lam1, lam2, lam3 = F.diagonal()
        q1y, q2x, q3z = q
        
        # residual
        R = tkd_residual.Equations(self.mu, self.fo, self.d, self.alpha, q1y, q2x, q3z, lam1, lam2, lam3, pressure)
        
        return R
    
    def evalJacobian(self, q, F, pressure):
        
        # stretches
        lam1c, lam2c, lam3c = F.diagonal()
        q1y, q2x, q3z = q
        
        # jacobian
        J = tkd_jacobian.Equations(self.mu, self.fo, self.d, self.alpha, q1y, q2x, q3z, lam1c, lam2c, lam3c, pressure)
        
        return J
    
    def averageStresses(self, F, pressure):

        """ It returns the averaged stresses in the TKD. Here, it's used a direct
        evaluation of quantities, nevertheless, a more detailed process is comented
        """

        # unit celll
        J = np.linalg.det(F)
        Vt = J * self.Vo

        # axial stresses
        self.fvec = self.strutsForces(self.A, self.lam, self.direc, pressure)
        avgsig = 2. / Vt * ( np.outer(self.fvec[6,:], np.dot(F, self.bx)) + 
                                   np.outer(self.fvec[9,:], np.dot(F, self.by)) + 
                                   np.outer(self.fvec[14,:], np.dot(F, self.bz)) )
                                  
        # avgsig = np.zeros([3, 3])
        # for i in np.arange(3):
            # avgsig[i, i] = 2. / Vt * F[i, i] * (self.fvec[6, i] * self.bx[i] + self.fvec[9, i] * self.by[i] + self.fvec[14, i] * self.bz[i])

        # terms of stresses associated to pressure
        # eff_length_rel = self.strutsEffectiveLengthFactor()
        # nelem = np.shape(self.lam)[0]
        # for i in np.arange(nelem):
            # A1 = np.outer(self.a1_direc[i,:], self.a1_direc[i,:])
            # A2 = np.outer(self.a2_direc[i,:], self.a2_direc[i,:])
            # avgsig = avgsig + 1./Vt*( -pressure * ( 2.- np.sqrt(self.lam[i]) ) * self.A[i]* (self.L[i] *  eff_length_rel[i]) * (A1+A2) )


        return avgsig
    
    def averageMaterialStress(self, F, pressure):
        
        """ It returns the averaged stresses in the TKD. Here, it's used a direct
        evaluation of quantities, nevertheless, a more detailed process is comented
        """

        # axial stresses
        self.fvec = self.strutsForces(self.A, self.lam, self.direc, pressure)
        
        # S = 2./Vto * np.dot( np.linalg.inv(F), np.outer(self.fvec[6,:], bx) + \
                              # np.outer(self.fvec[9,:], by) + np.outer(self.fvec[14,:], bz))

        S = np.zeros([3, 3])
        for i in np.arange(3):
            S[i, i] = 2. / self.Vo * 1./(F[i, i]) * (self.fvec[6, i] * self.bx[i] + 
                                                 self.fvec[9, i] * self.by[i] + 
                                                 self.fvec[14, i] * self.bz[i])

        return S
        
    
    def solveTKD(self, F, pressure):

        # find degrees of freedom by minimizing potential energy
        qo = np.zeros(3)
        [q, infodict, ier, msg] = fsolve(self.evalResidual, qo, args=(F, pressure), fprime=self.evalJacobian, full_output=True, xtol=1.e-13)
        # [q, infodict, ier, msg] = fsolve(self.evalResidual, qo, args=(F, pressure), full_output=True, xtol=1.e-13)
        print(msg + ' ' + str(infodict['nfev']) + ' iterations were necessary')

        # # set information ---------------------------------------------
        # current config
        self.setDeformedGeometry(q, F)
        
        # set energies
        self.setAxialEnergy()
        self.setRotationalEnergy()
        self.setExternalEnergy(pressure)
        self.Etot = self.Eaxial_tot + self.Erot_tot - self.Eext_tot

        # # Useful for averaging stresses
        self.direc = self.strutsDirections(self.xyz)
        self.a1_direc, self.a2_direc = self.strutsOrthogonalDirections(self.direc)

        # # stress average -----------------------------------------------
        avgsig = self.averageStresses(F, pressure)

        return q, avgsig
    
    
    def setAllData(self, q, F, pressure):
        
        # current config
        self.setDeformedGeometry(q, F)
        
        # set energies
        self.setAxialEnergy()
        self.setRotationalEnergy()
        self.setExternalEnergy(pressure)
        self.Etot = self.Eaxial_tot + self.Erot_tot - self.Eext_tot

        # # Useful for averaging stresses
        self.direc = self.strutsDirections(self.xyz)
        self.a1_direc, self.a2_direc = self.strutsOrthogonalDirections(self.direc)

    


    









class Utilities:

    def __init__(self):
        pass

    @staticmethod
    def writeEquations_txt(f, sn, path='./'):

        """
        write symbolic equations given by f in a text file
        f: list of matrix or vectorial equations to be written in path
        sn: symbolic name, example: dVdq[0]
        """
        neq = len(f)
        g = open(path, 'w')
        for k in np.arange(neq):

            # scalar
            if np.ndim(f[k]) == 0:
                g.write(sn[k] + ' = ' + str(f[k]) + '\n')
                g.write('\n \n')

            # vector
            elif np.ndim(f[k]) == 1:
                n = np.shape(f[k])[0]
                for i in np.arange(n):
                    print('Writing vector, coordinate '+str(i))
                    str_data = str(f[k][i])
                    str_data = Utilities.replaceFractions(str_data)
                    g.write(sn[k] + '[' + str(i) + ']' + ' = ' + str_data + '\n')
                    g.write('\n \n')

            # matrix
            elif np.ndim(f[k]) == 2:
                ni, nj = np.shape(f[k])
                for i in np.arange(ni):
                    for j in np.arange(nj):
                        print('Writing matrix, coordinates: ('+str(i)+','+str(j)+')')
                        str_data = str(f[k][i, j])
                        str_data = Utilities.replaceFractions(str_data)
                        g.write(sn[k] + '[' + str(i) + ',' + str(j) + ']' + ' = ' + str_data + '\n')
                        g.write('\n \n')

            else:
                raise Exception('UE: symbolic array dimension does not match')

        g.close()

    @staticmethod
    def replaceFractions(string):

        newString = string
        newString = newString.replace('0.166666666666667', 'one6')
        newString = newString.replace('0.333333333333333', 'one3')
        newString = newString.replace('0.666666666666667', 'two3')
        newString = newString.replace('1.33333333333333',  'four3')

        return newString

    @staticmethod
    def symCrossProduct(a, b):
        
        """
        a, b: symbolic arrays, i.e., a=np.array([a1, a2, a3]) where
        a_i is symbolic and a,b are numpy arrays
        """
        a1, a2, a3 = a
        b1, b2, b3 = b
        
        c = np.array([a2*b3-a3*b2, -a1*b3+a3*b1, a1*b2-a2*b1])
        return c
      
    @staticmethod
    def writeEquations_py(f,sn,var,varout,path):   
    
        """
        It writes symbolic expressions given by f in a python file
        f: equations (list)
        sn: symbolic name of each equation (list)
        var: function variables (string)
        varout: output variables (string)
        """
        
        g = open(path,'w')
        tab = '     '
        g.write('# -*- coding: utf-8 -*- \n')
        g.write('""" \n')
        g.write('""" \n')
        g.write('\n')    
        g.write('from __future__ import division \n')
        g.write('import numpy as np \n')
        g.write('from numpy import sqrt \n')
        
        # define constants
        g.write('one6  = 1./6. \n')
        g.write('one3  = 1./3. \n')
        g.write('two3  = 2./3. \n')
        g.write('four3 = 4./3. \n')

        # obtain variables
        g.write('def Equations('+var+'): \n') 
        neq = len(f)
        for k in np.arange(neq):
            
            # scalar
#            if f[k] == float:
#                g.write(tab + sn[k]+' = '+str(f[k])+'\n')
                
            # vector
            if len(f[k].shape) == 1:
                n = np.shape(f[k])[0]
                g.write(tab + sn[k] + '=np.zeros('+str(n)+') \n')
                for i in np.arange(n):
                    s = Utilities.replaceFractions(str(f[k][i]))
                    g.write(tab + sn[k]+'['+str(i)+']'+' = ' + s + '\n')
                    
            # matrix
            elif len(f[k].shape) == 2:
                ni,nj = np.shape(f[k])
                g.write(tab + sn[k] + '=np.zeros(['+str(ni)+','+str(nj)+']) \n')
                for i in np.arange(ni):
                    for j in np.arange(nj):
                        s = Utilities.replaceFractions(str(f[k][i,j]))
                        g.write(tab + sn[k]+'['+str(i)+','+str(j)+']'+' = ' + s + '\n')
            
        g.write(tab + 'return '+varout)
        g.close()



