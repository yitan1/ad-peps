""" 2D ising model """

import jax.numpy as np

from .common import sigmaz, sigmax, id2
from adpeps.utils.tlist import set_pattern
import adpeps.ipeps.config as sim_config

name = "ising spin-1/2 model"

def setup():
    """ Returns the Hamiltonian """
    H = make_hamiltonian(**sim_config.model_params)

    obs  = None
    return H, obs

def make_hamiltonian(h=1):
    """ ising model """
    H = -tprod(sigmax, sigmax) / 4 * 2 - \
            h* tprod(sigmaz, id2) / 2 / 2 - \
            h* tprod(id2, sigmaz) / 2 / 2
    return [H, H]

def tprod(a,b):
    return np.outer(a,b).reshape([2,2,2,2], order='F').transpose([0,2,1,3])