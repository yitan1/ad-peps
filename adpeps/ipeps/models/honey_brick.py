""" 2D ising model """

import jax.numpy as np

from .common import sigmaz, sigmax, sigmay, id2
from adpeps.utils.tlist import set_pattern
import adpeps.ipeps.config as sim_config

name = "brickwall kitaev spin-1/2 model"

def setup():
    """ Returns the Hamiltonian """
    H = make_hamiltonian(**sim_config.model_params)

    obs  = None
    return H, obs

def make_hamiltonian(Jx = 1, Jy = 1):
    """ kiteav model """

    Hh = Jx*tprod(sigmax, sigmax) / 4 + \
            Jy* tprod(sigmay, sigmay) / 4 
    Hv = tprod(sigmaz, sigmaz) / 4 
    return [Hh, Hv]

def tprod(a,b):
    return np.outer(a,b).reshape([2,2,2,2], order='F').transpose([0,2,1,3])