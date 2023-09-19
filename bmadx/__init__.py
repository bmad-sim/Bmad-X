# Constants
from .constants import *

# Structures (particle and elements named tuples)
from .structures import Particle, Drift, Quadrupole, CrabCavity, RFCavity, SBend, \
    Sextupole

# Tracking routines
from .tracking_routines.track_a_drift import make_track_a_drift
from .tracking_routines.track_a_quadrupole import make_track_a_quadrupole
from .tracking_routines.track_a_crab_cavity import make_track_a_crab_cavity
from .tracking_routines.track_a_rf_cavity import make_track_a_rf_cavity
from .tracking_routines.track_a_bend import make_track_a_bend
from .tracking_routines.track_a_sextupole import make_track_a_sextupole

# Tracking library dictionary (as of now, supports NumPy and PyTorch): 
import numpy as np
import torch

def make_tracking_dict(lib):
    dic = {
        "Drift" : make_track_a_drift(lib),
        "Quadrupole" : make_track_a_quadrupole(lib),
        "CrabCavity" : make_track_a_crab_cavity(lib),
        "RFCavity" : make_track_a_rf_cavity(lib),
        "SBend": make_track_a_bend(lib),
        "Sextupole": make_track_a_sextupole(lib)
    }
    return dic

LIB_DICT = {
    np: {
        'tracking_routine': make_tracking_dict(np),
        'number_type': np.ndarray,
        'construct_type': np.array
    },
    torch: {
        'tracking_routine': make_tracking_dict(torch),
        'number_type': torch.Tensor,
        'construct_type': torch.tensor
    }
}

# Tracking functions
from .track import track_element, track_lattice, track_lattice_save_particles, track_lattice_save_stats