# Structures (particle coords and elements)
from bmadx.bmad.modules import Particle, Drift, Quadrupole, CrabCavity, RFCavity, SBend

# Tracking routines
from bmadx.bmad.low_level.track_a_drift import make_track_a_drift
from bmadx.bmad.low_level.track_a_quadrupole import make_track_a_quadrupole
from bmadx.bmad.low_level.track_a_crab_cavity import make_track_a_crab_cavity
from bmadx.bmad.low_level.track_a_rf_cavity import make_track_a_rf_cavity
from bmadx.bmad.low_level.track_a_bend import make_track_a_bend

# Constants
from bmadx.sim_utils.interfaces.constants import *

# Tracking library dictionary: 
import numpy as np
import torch

def make_tracking_dict(lib):
    dic = {
        "Drift" : make_track_a_drift(lib),
        "Quadrupole" : make_track_a_quadrupole(lib),
        "CrabCavity" : make_track_a_crab_cavity(lib),
        "RFCavity" : make_track_a_rf_cavity(lib),
        "SBend": make_track_a_bend(lib),
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

# tracking functions
from bmadx.bmadx_utils.track import track_element, track_lattice, track_lattice_save_particles, track_lattice_save_stats