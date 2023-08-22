""" 2D ising model """

import jax.numpy as np
from adpeps.tensor.contractions import ncon

from .common import sigmaz, sigmax, sigmay, id2
from adpeps.utils.tlist import set_pattern
import adpeps.ipeps.config as sim_config

name = "honeycomb kitaev spin-1/2 model"

def setup():
    """ Returns the Hamiltonian """
    H = make_hamiltonian(**sim_config.model_params)

    # x = (tprod(id2, sigmax) + tprod(sigmax, id2)) /2 /2
    # y = (tprod(id2, sigmay) + tprod(sigmay, id2)) /2 /2
    # z = (tprod(id2, sigmaz) + tprod(sigmaz, id2)) /2 /2
    x = tprod(sigmax, sigmax)/4
    y = tprod(sigmay, sigmay)/4
    z = tprod(sigmaz, sigmaz)/4
    obs = [x, y, z]
    return H, obs

def make_hamiltonian(Jx = 1, Jy = 1, Jz = 1):
    """ kiteav model """
    
    Hh = Jx*tprod4(tprod(id2, sigmax) , tprod(sigmax, id2)) / 4  + \
            Jz*tprod4(tprod(sigmaz, sigmaz) , tprod(id2, id2)) / 4 / 2 / 2 + \
            Jz*tprod4(tprod(id2, id2), tprod(sigmaz, sigmaz)) / 4 / 2 / 2
    Hv = Jy*tprod4(tprod(id2, sigmay) , tprod(sigmay, id2)) / 4  + \
            Jz*tprod4(tprod(sigmaz, sigmaz) , tprod(id2, id2)) / 4 / 2 / 2 + \
            Jz*tprod4(tprod(id2, id2), tprod(sigmaz, sigmaz)) / 4 / 2 /2
    return [-Hh, -Hv]

def make_init_gs(D=2):
    Q_op = np.zeros((2,2,2,2,2), dtype = complex)
    Q_op = Q_op.at[:,:,0,0,0].set(id2)
    Q_op = Q_op.at[:,:,0,1,1].set(sigmax)
    Q_op = Q_op.at[:,:,1,0,1].set(sigmay)
    Q_op = Q_op.at[:,:,1,1,0].set(sigmaz)
    # print(type(Q_op))
    ux, uy, uz = 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)
    s111 = 1/np.sqrt(2+2*uz)*np.array([1 + uz, complex(ux,uy)])
    l = ncon([Q_op, s111], ([-1,1,-2,-3,-4], [1]))
    r = ncon([Q_op, s111], ([-1,1,-4,-3,-2], [1]))
    A = ncon([l, r], ([-1,-3,-4, 1], [-2, 1, -5,-6]))
    dimA = A.shape
    A = A.reshape(dimA[0]*dimA[1], dimA[2], dimA[3], dimA[4], dimA[5])
    
    if D == 4:
        phi = 0.24*np.pi
        a = np.tan(phi)
        R_op = np.zeros((2,2,2,2,2), dtype = complex)
        R_op = R_op.at[:,:,0,0,0].set(id2)
        R_op = R_op.at[:,:,0,1,1].set(sigmax*a)
        R_op = R_op.at[:,:,1,0,1].set(sigmay*a)
        R_op = R_op.at[:,:,1,1,0].set(sigmaz*a)
        RR = ncon([R_op, R_op], ([-1,-3,-5,-6,1],[-2,-4,-8,-7,1]))
        dRR = RR.shape
        RR = RR.reshape(dRR[0]*dRR[1], dRR[2]*dRR[3], dRR[4], dRR[5], dRR[6], dRR[7])
        A1 = ncon([RR, A], ([-1,1, -2, -4, -6, -8], [1, -3, -5, -7, -9]))
        D1 = A1.shape[1]
        D2 = A1.shape[2]
        A1 = A1.reshape(A1.shape[0], D1*D2, D1*D2, D1*D2, D1*D2)
        return A1.reshape(A1.size)
    return A.reshape(A.size)

def tprod(a,b):
    return np.outer(a,b).reshape([2,2,2,2], order='F').transpose([0,2,1,3]).reshape(4,4)

def tprod4(a,b):
    return np.outer(a,b).reshape([4,4,4,4], order='F').transpose([0,2,1,3])
