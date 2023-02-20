""" 2D ising model """

import jax.numpy as np

from .common import sigmaz, sigmax, sigmay, id2
from adpeps.utils.tlist import set_pattern
import adpeps.ipeps.config as sim_config

name = "honeycomb kitaev spin-1/2 model"

def setup():
    """ Returns the Hamiltonian """
    H = make_hamiltonian(**sim_config.model_params)

    obs  = None
    return H, obs

def make_hamiltonian(Jx = 1, Jy = 1):
    """ kiteav model """
    
    Hh = Jx*tprod4(tprod(id2, sigmax) , tprod(sigmax, id2)) / 4 + \
            tprod4(tprod(sigmaz, sigmaz) , tprod(id2, id2)) / 4 / 2 + \
            tprod4(tprod(id2, id2), tprod(sigmaz, sigmaz)) / 4 / 2
    Hv = Jy*tprod4(tprod(id2, sigmay) , tprod(sigmay, id2)) / 4 + \
            tprod4(tprod(sigmaz, sigmaz) , tprod(id2, id2)) / 4 / 2 + \
            tprod4(tprod(id2, id2), tprod(sigmaz, sigmaz)) / 4 / 2
    return [Hh, Hv]

def tprod(a,b):
    return np.outer(a,b).reshape([2,2,2,2], order='F').transpose([0,2,1,3]).reshape(4,4)

def tprod4(a,b):
    return np.outer(a,b).reshape([4,4,4,4], order='F').transpose([0,2,1,3])